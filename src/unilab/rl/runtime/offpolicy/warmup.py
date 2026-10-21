"""Learner-owned cold-path preparation contract for off-policy training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import torch


class DeviceRngStateApi(Protocol):
    """Per-device Torch RNG state operations used by warmup snapshots."""

    def get(self, device: torch.device) -> torch.Tensor: ...

    def set(self, state: torch.Tensor, device: torch.device) -> None: ...

    def state_available(self, device: torch.device) -> bool: ...


class CudaRngStateApi:
    """CUDA state API; snapshots require an initialized context."""

    def get(self, device: torch.device) -> torch.Tensor:
        return torch.cuda.get_rng_state(device)

    def set(self, state: torch.Tensor, device: torch.device) -> None:
        torch.cuda.set_rng_state(state, device)

    def state_available(self, device: torch.device) -> bool:
        del device
        return bool(torch.cuda.is_initialized())


class MpsRngStateApi:
    """MPS state API; the public interface has no device index."""

    def get(self, device: torch.device) -> torch.Tensor:
        del device
        return torch.mps.get_rng_state()

    def set(self, state: torch.Tensor, device: torch.device) -> None:
        del device
        torch.mps.set_rng_state(state)

    def state_available(self, device: torch.device) -> bool:
        del device
        return bool(torch.backends.mps.is_available())


_RNG_STATE_APIS: dict[str, DeviceRngStateApi] = {
    "cuda": CudaRngStateApi(),
    "mps": MpsRngStateApi(),
}


@dataclass(frozen=True)
class OffPolicyWarmupContext:
    """Representative shapes and lifecycle options for no-side-effect warmup.

    The inference tensors are learner-device scratch allocations. When the
    runtime provides ``replay_batch``, those views alias a replay pipeline's
    scratch cold slot to avoid allocating a third full training batch during
    warmup. They are deliberately
    synthetic: constructing an environment or consuming replay data during
    preparation would change collection and training semantics.
    """

    inference_observations: torch.Tensor
    inference_dones: torch.Tensor
    batch_size: int
    updates_per_step: int
    policy_frequency: int
    target_frequency: int
    policy_before_critic: bool
    replay_batch: dict[str, torch.Tensor] | None = None


class OffPolicyLearnerPreparation(Protocol):
    """Optional owner-layer hook invoked before the collector subprocess starts."""

    def prepare_for_collection(self, warmup_context: OffPolicyWarmupContext) -> None:
        """Prepare learner-owned cold paths without changing training state."""
        ...


@dataclass
class _TorchRngState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None
    mps: torch.Tensor | None
    python: object
    device: torch.device


def capture_rng_state(device: str | torch.device) -> _TorchRngState:
    torch_device = torch.device(device)
    rng_api = _RNG_STATE_APIS.get(torch_device.type)
    device_state = (
        rng_api.get(torch_device)
        if rng_api is not None and rng_api.state_available(torch_device)
        else None
    )
    return _TorchRngState(
        cpu=torch.random.get_rng_state(),
        cuda=device_state if torch_device.type == "cuda" else None,
        mps=device_state if torch_device.type == "mps" else None,
        python=random.getstate(),
        device=torch_device,
    )


def restore_rng_state(state: _TorchRngState) -> None:
    torch.random.set_rng_state(state.cpu)
    device_state = state.cuda if state.cuda is not None else state.mps
    rng_api = _RNG_STATE_APIS.get(state.device.type)
    if device_state is not None and rng_api is not None:
        rng_api.set(device_state, state.device)
    random.setstate(state.python)  # type: ignore[arg-type]


def build_offpolicy_warmup_batch(
    context: OffPolicyWarmupContext,
    *,
    obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Build a representative replay batch without touching replay state."""
    rows = max(1, int(context.batch_size) * max(1, int(context.updates_per_step)))
    torch_device = torch.device(device)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=torch_device)

    return {
        "obs": tensor(rows, obs_dim),
        "actions": tensor(rows, action_dim),
        "rewards": tensor(rows),
        "next_obs": tensor(rows, obs_dim),
        "dones": tensor(rows),
        "truncated": tensor(rows),
        "critic": tensor(rows, critic_obs_dim),
        "next_critic": tensor(rows, critic_obs_dim),
    }


__all__ = [
    "CudaRngStateApi",
    "DeviceRngStateApi",
    "MpsRngStateApi",
    "OffPolicyLearnerPreparation",
    "OffPolicyWarmupContext",
    "capture_rng_state",
    "build_offpolicy_warmup_batch",
    "restore_rng_state",
]
