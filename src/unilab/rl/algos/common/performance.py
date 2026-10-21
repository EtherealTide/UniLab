"""Learner performance tiers shared by the off-policy SAC family.

Entries in this module are **validated performance policies**, not runtime
capability probes.  They deliberately do not try-and-degrade: for example,
ROCm graph capture may succeed yet replay incorrectly, so capture success is
not evidence that the accelerated learner tier can be enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class LearnerPerformanceTier:
    """A human-validated learner performance tier."""

    name: str
    device_type: str
    vendor_runtime: str | None
    supports_whole_cycle_graph: bool
    requires_inductor: bool
    supports_capturable_optimizer: bool
    evidence: Literal["policy"] = "policy"

    def manifest(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "device_type": self.device_type,
            "vendor_runtime": self.vendor_runtime or "default",
            "supports_whole_cycle_graph": self.supports_whole_cycle_graph,
            "requires_inductor": self.requires_inductor,
            "supports_capturable_optimizer": self.supports_capturable_optimizer,
            "evidence": self.evidence,
        }


_NVIDIA_CUDA_TIER = LearnerPerformanceTier(
    name="nvidia_cuda",
    device_type="cuda",
    vendor_runtime="nvidia",
    supports_whole_cycle_graph=True,
    requires_inductor=True,
    supports_capturable_optimizer=True,
)
_COMPATIBILITY_TIERS = {
    device_type: LearnerPerformanceTier(
        name=f"{device_type}_compatibility",
        device_type=device_type,
        vendor_runtime=None,
        supports_whole_cycle_graph=False,
        requires_inductor=False,
        supports_capturable_optimizer=False,
    )
    for device_type in ("cuda", "mps", "xpu", "cpu")
}


def resolve_learner_performance_tier(device: str | torch.device) -> LearnerPerformanceTier:
    """Resolve the validated learner policy for ``device``.

    A CUDA device backed by the HIP runtime is always the compatibility tier:
    that exclusion is a policy decision, not a claim that every ROCm operation
    is impossible.
    """
    try:
        resolved = torch.device(device)
    except RuntimeError as exc:
        raise ValueError("SAC and FlashSAC learners support CUDA, MPS, XPU, and CPU only") from exc
    if resolved.type not in _COMPATIBILITY_TIERS:
        raise ValueError(
            f"SAC and FlashSAC learners support CUDA, MPS, XPU, and CPU only; got {resolved.type!r}"
        )
    if resolved.type == "cuda":
        torch_version = getattr(torch, "version", None)
        if not getattr(torch_version, "hip", None):
            return _NVIDIA_CUDA_TIER
        tier = _COMPATIBILITY_TIERS["cuda"]
        return LearnerPerformanceTier(
            name="rocm_cuda_compatibility",
            device_type=tier.device_type,
            vendor_runtime="rocm",
            supports_whole_cycle_graph=tier.supports_whole_cycle_graph,
            requires_inductor=tier.requires_inductor,
            supports_capturable_optimizer=tier.supports_capturable_optimizer,
        )
    return _COMPATIBILITY_TIERS[resolved.type]
