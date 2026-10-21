"""Pure collector-tensor placement resolution for the off-policy runtime.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.tensor_runtime``. These functions never read a Hydra config;
config-level resolution lives in :mod:`unilab.training.tensor_runtime`.

Placement is one-dimensional: ``training.collector_tensor_device``
(auto/cpu/cuda) decides where collector-facing tensors reside, and the
inference transport (host ring vs CUDA ring) is a derived fact of that
decision — there is no independent transport configuration. An explicit
``cpu`` request with a CUDA learner fails closed unless the run carries the
attribution-baseline/debug opt-in (issue #2483).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from unilab.process_context import CUDA_LEARNER_CPU_CARRIER_OPT_IN


class InferenceTransport(str, Enum):
    """The two ring/scratch topologies the runtime promises to implement."""

    CUDA = "cuda"
    HOST = "host"


@dataclass(frozen=True)
class InferencePlacement:
    """Resolved collector/ring/env/learner placement before spawn.

    ``staging_policy`` is one of ``cuda_no_host_boundary`` (CUDA transport),
    ``cpu_ring_explicit_learner_actor_h2d_action_d2h`` (host transport, CUDA
    learner), ``cpu_ring_explicit_learner_device_copy`` (host transport,
    non-CUDA accelerator learner such as ``torch.mps`` (Metal Performance
    Shaders)), or ``cpu_no_device_transfer`` (host transport, CPU learner).
    """

    mode: InferenceTransport
    env_device: str
    ring_device: str
    learner_device: str
    collector_tensor_native: bool
    staging_policy: str

    @property
    def transport_spec(self) -> "InferenceTransportSpec":
        return inference_transport_spec(self.mode)

    def manifest(self) -> dict[str, str | dict[str, str | bool]]:
        return {
            "mode": self.mode.value,
            "env_device": self.env_device,
            "ring_device": self.ring_device,
            "learner_device": self.learner_device,
            "staging_policy": self.staging_policy,
            "transport_table": self.transport_spec.manifest(),
        }


@dataclass(frozen=True)
class InferenceTransportSpec:
    """Placement-level topology contract for one inference transport mode."""

    mode: InferenceTransport
    ring_device_type: str
    requires_tensor_native: bool
    requires_rank_local_device_match: bool
    publication_sync: str

    def manifest(self) -> dict[str, str | bool]:
        return {
            "ring_device_type": self.ring_device_type,
            "requires_tensor_native": self.requires_tensor_native,
            "requires_rank_local_device_match": self.requires_rank_local_device_match,
            "publication_sync": self.publication_sync,
            "evidence": "policy",
        }


_INFERENCE_TRANSPORT_SPECS: dict[InferenceTransport, InferenceTransportSpec] = {
    InferenceTransport.CUDA: InferenceTransportSpec(
        mode=InferenceTransport.CUDA,
        ring_device_type="cuda",
        requires_tensor_native=True,
        requires_rank_local_device_match=True,
        publication_sync="cuda_ipc_events",
    ),
    InferenceTransport.HOST: InferenceTransportSpec(
        mode=InferenceTransport.HOST,
        ring_device_type="cpu",
        requires_tensor_native=False,
        requires_rank_local_device_match=False,
        publication_sync="cpu_synchronous",
    ),
}


def inference_transport_spec(mode: InferenceTransport) -> InferenceTransportSpec:
    """Return the authoritative topology table entry for ``mode``."""
    return _INFERENCE_TRANSPORT_SPECS[mode]


def _canonical_device(value: str, *, label: str, algo_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{algo_name} {label} must be a non-empty string, got {value!r}")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{algo_name} {label} is not a Torch device: {value!r}") from exc
    if device.type == "cuda" and device.index is None:
        try:
            return f"cuda:{torch.cuda.current_device()}"
        except (AssertionError, RuntimeError):
            # CPU-only test/CI builds can parse CUDA requests before a fake
            # device is injected; preserve the explicitly requested topology.
            return "cuda"
    return str(device)


def _device_requires_cuda(device: str) -> bool:
    return str(torch.device(device).type) == "cuda"


def _torch_reports_hip_build() -> bool:
    """Whether this Torch build presents a ROCm/HIP CUDA runtime."""
    return bool(getattr(torch.version, "hip", None))


def cpu_ring_staging_policy(learner_device: str) -> str:
    """Return the CPU-ring staging policy for one canonical learner device."""
    device_type = torch.device(learner_device).type
    policies = {
        "cpu": "cpu_no_device_transfer",
        "cuda": "cpu_ring_explicit_learner_actor_h2d_action_d2h",
    }
    if device_type in policies:
        return policies[device_type]
    # Other accelerators (e.g. torch.mps (Metal Performance Shaders)) still
    # cross devices at the CPU-ring copy_ boundary.
    return "cpu_ring_explicit_learner_device_copy"


def resolve_inference_placement(
    *,
    learner_device: str,
    collector_tensor_device: str | None = None,
    algo_name: str,
    allow_cpu_carrier_with_cuda_learner: bool = False,
) -> InferencePlacement:
    """Resolve one explicit topology and reject mixed placement before spawn.

    ``collector_tensor_device`` is the sole placement request: ``cpu`` keeps
    collector-facing tensors on the host (CPU ring), ``cuda`` places them on
    the learner's CUDA device (CUDA ring), and ``auto``/``None`` follows the
    learner device (CUDA learner → CUDA, any other learner → CPU). The
    inference transport is derived from that request; it has no independent
    configuration. Host-ring staging policies are selected by
    ``cpu_ring_staging_policy`` from the learner device type.

    An explicit ``cpu`` request with a CUDA learner fails closed (issue
    #2483): the CPU carrier survives only for CPU/torch.mps learners and
    ROCm/HIP builds, where it remains the platform fallback. Attribution
    baselines and debug runs opt in by passing
    ``allow_cpu_carrier_with_cuda_learner=True`` (the training config layer
    binds that flag to the ``UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER``
    environment variable).
    """
    learner = _canonical_device(learner_device, label="learner device", algo_name=algo_name)
    if collector_tensor_device is None:
        collector_tensor_device = "auto"
    if not isinstance(collector_tensor_device, str):
        raise TypeError(
            f"{algo_name} collector tensor device must be a string or omitted, "
            f"got {collector_tensor_device!r}"
        )
    requested = collector_tensor_device.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"{algo_name} collector tensor device must be one of {{'auto', 'cpu', 'cuda'}} "
            f"(null=auto), got {collector_tensor_device!r}"
        )
    if requested == "auto":
        requested = "cuda" if _device_requires_cuda(learner) else "cpu"

    if requested == "cuda":
        if not _device_requires_cuda(learner):
            raise ValueError(
                f"{algo_name} collector_tensor_device=cuda requires a CUDA learner "
                f"device; got learner={learner!r}"
            )
        return InferencePlacement(
            mode=InferenceTransport.CUDA,
            env_device=learner,
            ring_device=learner,
            learner_device=learner,
            collector_tensor_native=True,
            staging_policy="cuda_no_host_boundary",
        )

    if (
        _device_requires_cuda(learner)
        and not _torch_reports_hip_build()
        and not allow_cpu_carrier_with_cuda_learner
    ):
        raise ValueError(
            f"{algo_name} collector_tensor_device=cpu with CUDA learner {learner!r} "
            "is not supported: with a CUDA learner the CPU carrier is retained only "
            "for ROCm/HIP builds and non-CUDA accelerator learners (issue #2483). "
            "Use collector_tensor_device=auto/cuda, or export "
            f"{CUDA_LEARNER_CPU_CARRIER_OPT_IN}=1 for attribution-baseline/debug runs."
        )

    return InferencePlacement(
        mode=InferenceTransport.HOST,
        env_device="cpu",
        ring_device="cpu",
        learner_device=learner,
        collector_tensor_native=False,
        staging_policy=cpu_ring_staging_policy(learner),
    )
