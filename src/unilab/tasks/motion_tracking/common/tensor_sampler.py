"""Device-resident motion sampling for tensor Manager owners.

The community ``MotionSampler`` remains the NumPy reference implementation and
cold-path loader. This owner adapter owns the hot tensor lifecycle: adaptive
failure statistics, frame advance, reset sampling, and sampling diagnostics stay
on one Torch device and consume the Manager-owned Torch generator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


def _sampling_dispatch_kernel(
    rows: torch.Tensor,
    frames: torch.Tensor,
    clip_offsets: torch.Tensor,
    clip_end_frames: torch.Tensor,
    current_frames: torch.Tensor,
    current_clip_end_frames: torch.Tensor,
) -> None:
    # Keep this tiny selected-row dispatch eager. An Inductor fusion of
    # searchsorted/index-select measured no wall-time benefit at ~39 reset rows,
    # and synthetic wide-clip tensors triggered a Triton index assertion while
    # autotuning this graph. The surrounding eager operations are stable and
    # already device-resident.
    current_frames.index_copy_(0, rows, frames)
    clip_indices = torch.searchsorted(clip_offsets, frames.to(dtype=torch.int64), right=True)
    clip_indices.sub_(1).clamp_(min=0)
    current_clip_end_frames.index_copy_(0, rows, clip_end_frames.index_select(0, clip_indices))


@dataclass
class TensorMotionSamplerDiagnostics:
    """Counters for explicit device-to-host transfers caused by compatibility."""

    reset_host_row_transfers: int = 0
    step_host_mirror_transfers: int = 0

    @property
    def total(self) -> int:
        return self.reset_host_row_transfers + self.step_host_mirror_transfers


class TensorMotionSampler:
    """Torch-native adaptive/mixed motion sampler for ``TensorMotionCommand``."""

    def __init__(
        self,
        *,
        mode: str,
        num_envs: int,
        num_frames: int,
        clip_offsets: np.ndarray,
        clip_end_frames: np.ndarray,
        bin_count: int,
        adaptive_lambda: float,
        adaptive_kernel_size: int,
        adaptive_uniform_ratio: float,
        adaptive_alpha: float,
        start_ratio: float,
        initial_frames: np.ndarray,
        initial_clip_end_frames: np.ndarray,
        device: torch.device,
    ) -> None:
        if mode not in {"adaptive", "mixed"}:
            raise NotImplementedError(
                f"TensorMotionSampler supports adaptive/mixed modes only, got {mode!r}"
            )
        if num_envs <= 0 or num_frames <= 0 or bin_count <= 0:
            raise ValueError("TensorMotionSampler dimensions must be positive")
        if adaptive_kernel_size <= 0:
            raise ValueError("TensorMotionSampler adaptive_kernel_size must be positive")
        if not 0.0 <= adaptive_uniform_ratio or not 0.0 <= start_ratio <= 1.0:
            raise ValueError("TensorMotionSampler sampling ratios are outside [0, 1]")
        if adaptive_alpha < 0.0:
            raise ValueError("TensorMotionSampler adaptive_alpha must be non-negative")

        self.mode = mode
        self.num_envs = int(num_envs)
        self.num_frames = int(num_frames)
        self.bin_count = int(bin_count)
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_kernel_size = int(adaptive_kernel_size)
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_alpha = float(adaptive_alpha)
        self.start_ratio = float(start_ratio)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", index=torch.cuda.current_device())

        self.current_frames = torch.as_tensor(
            np.asarray(initial_frames, dtype=np.int32, copy=True), device=self.device
        )
        self.current_clip_end_frames = torch.as_tensor(
            np.asarray(initial_clip_end_frames, dtype=np.int32, copy=True), device=self.device
        )
        self._clip_offsets = torch.as_tensor(
            np.asarray(clip_offsets, dtype=np.int64, copy=True), device=self.device
        )
        self._clip_end_frames = torch.as_tensor(
            np.asarray(clip_end_frames, dtype=np.int32, copy=True), device=self.device
        )
        self._bin_failed = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
        kernel = np.asarray(
            [self.adaptive_lambda**index for index in range(self.adaptive_kernel_size)],
            dtype=np.float32,
        )
        kernel /= kernel.sum()
        self._adaptive_kernel = torch.as_tensor(kernel, device=self.device)
        self.sampling_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        self.sampling_top1_prob = torch.zeros((), dtype=torch.float32, device=self.device)
        self.sampling_top1_bin = torch.zeros((), dtype=torch.float32, device=self.device)
        self.diagnostics = TensorMotionSamplerDiagnostics()

    @property
    def bin_failed_count(self) -> torch.Tensor:
        return self._bin_failed

    def sample_frames(self, rows: torch.Tensor, torch_rng: torch.Generator | None) -> torch.Tensor:
        dispatch_started = time.perf_counter()
        if rows.ndim != 1 or rows.dtype != torch.int64 or rows.device != self.device:
            raise ValueError("TensorMotionSampler rows must be one-dimensional int64 device rows")
        if torch_rng is None:
            raise NotImplementedError("TensorMotionSampler requires the Manager Torch generator")
        count = rows.numel()
        frames: torch.Tensor
        probabilities: torch.Tensor | None = None
        if self.mode == "adaptive":
            probabilities = self._adaptive_probabilities()
            bins = torch.multinomial(probabilities, count, replacement=True, generator=torch_rng)
            offsets = torch.rand(count, device=self.device, generator=torch_rng)
            frames = (
                (bins.to(dtype=torch.float32) + offsets) / self.bin_count * (self.num_frames - 1)
            ).to(dtype=torch.int32)
        else:
            use_start = (
                torch.rand(count, device=self.device, generator=torch_rng) < self.start_ratio
            )
            uniform_frames = torch.randint(
                0,
                self.num_frames,
                (count,),
                device=self.device,
                generator=torch_rng,
                dtype=torch.int32,
            )
            frames = torch.where(use_start, torch.zeros_like(uniform_frames), uniform_frames)

        _sampling_dispatch_kernel(
            rows,
            frames,
            self._clip_offsets,
            self._clip_end_frames,
            self.current_frames,
            self.current_clip_end_frames,
        )
        if probabilities is not None:
            self._publish_sampling_metrics(probabilities)
        elif self.mode == "mixed":
            self._publish_mixed_metrics()
        self.last_reset_dispatch_ms = (time.perf_counter() - dispatch_started) * 1000.0
        return frames

    def update_failure_stats(self, terminated: torch.Tensor) -> None:
        if self.mode != "adaptive":
            return
        frames = self.current_frames.to(dtype=torch.int64)
        bin_indices = (frames * self.bin_count // max(self.num_frames, 1)).clamp_(
            max=self.bin_count - 1
        )
        discard_bin = self.bin_count
        failed_bins = torch.where(
            terminated, bin_indices, torch.full_like(bin_indices, discard_bin)
        )
        failures = torch.bincount(failed_bins, minlength=discard_bin + 1)[: self.bin_count].to(
            dtype=torch.float32
        )
        # Keep the all-failed/all-active scalar on-device. Converting ``any()``
        # to Python would synchronize at this point in every update-state pass.
        alpha = terminated.any().reshape(1).to(dtype=self._bin_failed.dtype) * self.adaptive_alpha
        self._bin_failed.mul_(1.0 - alpha).add_(failures * alpha)

    def step_full(self, active: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Advance and publish full-width frame mirrors without row gathers."""
        if active.shape != (self.num_envs,) or active.device != self.device:
            raise ValueError("TensorMotionSampler active mask shape/device mismatch")
        if (
            time_steps.shape != (self.num_envs,)
            or time_steps.dtype != torch.int32
            or time_steps.device != self.device
        ):
            raise ValueError("TensorMotionSampler time_steps must be int32 device frames")
        increment = active.to(dtype=torch.int32)
        self.current_frames.add_(increment)
        time_steps.copy_(self.current_frames)
        frames = self.current_frames.to(dtype=torch.int64)
        clip_indices = self._clip_indices(frames)
        clip_ends = self._clip_end_frames.index_select(0, clip_indices)
        self.current_clip_end_frames.copy_(clip_ends)
        done = frames > clip_ends
        done.logical_and_(active)
        return done.nonzero(as_tuple=False).flatten()

    def step(self, active: torch.Tensor, *, rows: torch.Tensor | None = None) -> torch.Tensor:
        if active.shape != (self.num_envs,) or active.device != self.device:
            raise ValueError("TensorMotionSampler active mask shape/device mismatch")
        selector = torch.arange(self.num_envs, device=self.device) if rows is None else rows
        if selector.ndim != 1 or selector.dtype != torch.int64 or selector.device != self.device:
            raise ValueError("TensorMotionSampler selected rows must be int64 device rows")
        selected_active = active.index_select(0, selector)
        self.current_frames.index_add_(0, selector, selected_active.to(dtype=torch.int32))
        frames = self.current_frames.to(dtype=torch.int64)
        selected_frames = frames.index_select(0, selector)
        clip_indices = self._clip_indices(selected_frames)
        clip_ends = self._clip_end_frames.index_select(0, clip_indices)
        self.current_clip_end_frames.index_copy_(0, selector, clip_ends)
        done = selected_frames > clip_ends
        done.logical_and_(selected_active)
        return done.nonzero(as_tuple=False).flatten()

    def _clip_indices(self, frames: torch.Tensor) -> torch.Tensor:
        indices = torch.searchsorted(
            self._clip_offsets, frames.to(dtype=torch.int64), right=True
        ).sub_(1)
        return indices.clamp_(min=0)

    def _adaptive_probabilities(self) -> torch.Tensor:
        probabilities = self._bin_failed + self.adaptive_uniform_ratio / self.bin_count
        if self.adaptive_kernel_size > 1:
            probabilities = torch.nn.functional.pad(
                probabilities, (0, self.adaptive_kernel_size - 1), mode="replicate"
            )
            probabilities = torch.nn.functional.conv1d(
                probabilities[None, None, :],
                self._adaptive_kernel[None, None, :],
                padding=0,
            )[0, 0]
        return probabilities / probabilities.sum()

    def _publish_sampling_metrics(self, probabilities: torch.Tensor) -> None:
        if self.bin_count > 1:
            entropy = -(probabilities * torch.log(probabilities + 1.0e-12)).sum()
            self.sampling_entropy.copy_(
                (entropy / torch.log(torch.tensor(float(self.bin_count), device=self.device)))
            )
        else:
            self.sampling_entropy.fill_(1.0)
        top1_index = torch.argmax(probabilities)
        self.sampling_top1_prob.copy_(probabilities[top1_index])
        self.sampling_top1_bin.copy_(top1_index.to(dtype=torch.float32) / float(self.bin_count))

    def _publish_mixed_metrics(self) -> None:
        start_mass = self.start_ratio + (1.0 - self.start_ratio) / self.bin_count
        self.sampling_top1_prob.fill_(start_mass)
        self.sampling_top1_bin.fill_(0.0)
        if start_mass >= 1.0 - 1.0e-9:
            self.sampling_entropy.fill_(0.0)
            return
        uniform_mass = (1.0 - self.start_ratio) / self.bin_count
        entropy = -start_mass * torch.log(torch.tensor(start_mass + 1.0e-12, device=self.device))
        entropy -= (
            (self.bin_count - 1)
            * uniform_mass
            * torch.log(torch.tensor(uniform_mass + 1.0e-12, device=self.device))
        )
        denominator = torch.log(torch.tensor(float(self.bin_count), device=self.device))
        self.sampling_entropy.copy_(
            entropy / denominator if self.bin_count > 1 else torch.tensor(1.0)
        )


__all__ = ["TensorMotionSampler", "TensorMotionSamplerDiagnostics"]
