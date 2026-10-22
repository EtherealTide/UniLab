"""TF32 backend selection resolved before learner construction.

This module is a leaf of :mod:`unilab.rl.runtime`: stdlib and torch only, no
unilab imports, so the off-policy runner and APPO runner can format policy
metadata without breaking the ``rl.runtime`` import DAG. Hydra config reading
lives in the training assembly layer.

TF32 is a meaningful win on the FP32 PPO/APPO update path; AMP-wrapped
SAC-family updates already cast their main matmuls, so that surface does not
expose this knob (see issue #2477's closed benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

DEFAULT_TF32_MATMUL = "ieee"
TF32_MATMUL_VALUES = ("ieee", "tf32")
_NVIDIA_TENSOR_CORE_MIN_COMPUTE_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class Tf32Policy:
    """The requested and resolved TF32 backend policy."""

    requested: str
    enabled: bool
    reason: str
    applied: bool = False

    def manifest(self) -> dict[str, str | bool]:
        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "reason": self.reason,
            "applied": self.applied,
        }


def normalize_tf32_matmul(value: Any) -> str:
    """Validate the user-facing ``training.tf32_matmul`` request.

    ``None`` keeps the owner default. Values are case-insensitive; anything
    outside :data:`TF32_MATMUL_VALUES` fails closed.
    """
    if value is None:
        return DEFAULT_TF32_MATMUL
    normalized = str(value).strip().lower()
    if normalized not in TF32_MATMUL_VALUES:
        valid = ", ".join(TF32_MATMUL_VALUES)
        raise ValueError(
            f"training.tf32_matmul must be one of {valid}; got {value!r}. "
            "TF32 only accelerates the FP32 PPO/APPO path; it is not exposed "
            "for AMP-wrapped SAC-family learners."
        )
    return normalized


def resolve_tf32_matmul(
    requested: Any,
    *,
    device: Any = None,
    torch_module: Any | None = None,
) -> Tf32Policy:
    """Resolve ``training.tf32_matmul`` into the effective backend policy.

    ``tf32`` enables TF32 only where it can execute (NVIDIA CUDA with compute
    capability >= 7.0) and falls back to IEEE on MPS/macOS, ROCm/HIP,
    pre-Ampere CUDA GPUs, and CPU. The fallback reason is recorded so runs
    stay auditable.
    """
    normalized = normalize_tf32_matmul(requested)
    if normalized == "ieee":
        return Tf32Policy(requested=normalized, enabled=False, reason="configured_ieee")

    torch_runtime = torch_module
    if torch_runtime is None:
        import torch as imported_torch

        torch_runtime = imported_torch

    device_type = str(torch_runtime.device(device).type if device is not None else "cpu")
    if device_type != "cuda":
        reason = "cpu" if device_type == "cpu" else f"unsupported_device_{device_type}"
        return Tf32Policy(requested=normalized, enabled=False, reason=reason)
    if bool(getattr(torch_runtime.version, "hip", None)):
        return Tf32Policy(requested=normalized, enabled=False, reason="rocm_hip")

    cuda = getattr(torch_runtime, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available) or not is_available():
        return Tf32Policy(requested=normalized, enabled=False, reason="cuda_unavailable")

    device_text = str(device)
    if device_text.startswith("cuda:") or device_text.startswith("cuda "):
        cuda.set_device(device)
    capability = cuda.get_device_capability(device)
    if capability < _NVIDIA_TENSOR_CORE_MIN_COMPUTE_CAPABILITY:
        return Tf32Policy(
            requested=normalized,
            enabled=False,
            reason="cuda_compute_capability_below_sm70",
        )
    return Tf32Policy(requested=normalized, enabled=True, reason="nvidia_cuda_tensor_cores")


def resolve_tf32_matmul_from_cfg(
    cfg: Any,
    *,
    device: Any = None,
    torch_module: Any | None = None,
) -> Tf32Policy:
    """Resolve the ``training.tf32_matmul`` owner config value."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        requested = OmegaConf.select(cfg, "training.tf32_matmul", default=None)
    else:
        training = cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)
        requested = (
            training.get("tf32_matmul")
            if isinstance(training, dict)
            else getattr(training, "tf32_matmul", None)
        )
    return resolve_tf32_matmul(requested, device=device, torch_module=torch_module)


def apply_tf32_matmul(
    requested: Any,
    *,
    device: Any = None,
    torch_module: Any | None = None,
) -> Tf32Policy:
    """Apply the resolved TF32 policy to the current torch runtime."""
    torch_runtime = torch_module
    if torch_runtime is None:
        import torch as imported_torch

        torch_runtime = imported_torch
    policy = resolve_tf32_matmul(requested, device=device, torch_module=torch_runtime)
    return _apply_policy(torch_runtime, policy)


def apply_tf32_matmul_from_cfg(
    cfg: Any,
    *,
    device: Any = None,
    torch_module: Any | None = None,
) -> Tf32Policy:
    """Resolve and apply the ``training.tf32_matmul`` owner value."""
    torch_runtime = torch_module
    if torch_runtime is None:
        import torch as imported_torch

        torch_runtime = imported_torch
    policy = resolve_tf32_matmul_from_cfg(cfg, device=device, torch_module=torch_runtime)
    return _apply_policy(torch_runtime, policy)


def _apply_policy(torch_runtime: Any, policy: Tf32Policy) -> Tf32Policy:
    backend = torch_runtime.backends
    value = "tf32" if policy.enabled else "ieee"
    matmul = backend.cuda.matmul
    cudnn = backend.cudnn
    if hasattr(matmul, "fp32_precision"):
        # torch >= 2.9: a new-API write flips the per-op TF32 split while the
        # legacy getter keeps its cached value, so any later legacy read (for
        # example torch.export's cudnn.set_flags) raises. Re-assert both
        # aliases first to keep the two APIs coherent.
        try:
            matmul.allow_tf32 = policy.enabled
            cudnn.allow_tf32 = policy.enabled
        except (AttributeError, TypeError, RuntimeError):
            pass
        try:
            matmul.fp32_precision = value
            cudnn.fp32_precision = value
        except (AttributeError, TypeError, RuntimeError):
            matmul.allow_tf32 = policy.enabled
            cudnn.allow_tf32 = policy.enabled
    else:
        # torch < 2.9 exposes the same knob through the boolean alias only.
        matmul.allow_tf32 = policy.enabled
        cudnn.allow_tf32 = policy.enabled
    return replace(policy, applied=True)


def format_tf32_matmul(policy: Tf32Policy | None) -> str:
    """Format the resolved policy for learner startup status logging."""
    if policy is None:
        return "TF32 matmul: unspecified"
    state = "tf32" if policy.enabled else "ieee"
    return f"TF32 matmul: {state} ({policy.reason})"
