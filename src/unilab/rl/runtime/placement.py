"""Pure placement resolution for the explicit off-policy inference transport.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.tensor_runtime``. These functions never read a Hydra config;
config-level resolution lives in :mod:`unilab.training.tensor_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


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
    transport: str | None = None,
    env_tensor_native: bool | None = None,
    algo_name: str,
) -> InferencePlacement:
    """Resolve one explicit topology and reject mixed placement before spawn.

    ``transport`` is authoritative when supplied. ``env_tensor_native`` is the
    owner's probe-derived carrier fact; combining tensor-native observations
    with host transport is incoherent and fails closed. Host-transport staging
    policies are selected by ``cpu_ring_staging_policy`` from the learner device
    type.
    """
    learner = _canonical_device(learner_device, label="learner device", algo_name=algo_name)
    if env_tensor_native is None and transport == InferenceTransport.CUDA.value:
        env_tensor_native = True
    if env_tensor_native is None:
        env_tensor_native = False
    requested_env_device = learner if env_tensor_native else "cpu"
    env_resolved = _canonical_device(requested_env_device, label="env device", algo_name=algo_name)

    if env_tensor_native and transport is None:
        transport = InferenceTransport.CUDA.value
    elif transport is None:
        transport = InferenceTransport.HOST.value
    if not isinstance(transport, str):
        raise TypeError(f"{algo_name} inference transport must be a string, got {transport!r}")
    transport = transport.strip().lower()
    if transport not in {item.value for item in InferenceTransport}:
        raise ValueError(
            f"{algo_name} inference transport must be one of {{'cuda', 'host'}} "
            f"(null=auto); use 'host' for the CPU inference ring, got {transport!r}"
        )

    if env_tensor_native and transport == InferenceTransport.HOST.value:
        raise ValueError(
            f"{algo_name} requests CUDA tensor-native env observations with host inference transport"
        )

    if transport == InferenceTransport.CUDA.value:
        if not _device_requires_cuda(learner) or not _device_requires_cuda(env_resolved):
            raise ValueError(
                f"{algo_name} CUDA inference transport requires CUDA env and learner "
                f"devices; got env={env_resolved!r}, learner={learner!r}"
            )
        if learner != env_resolved:
            raise ValueError(
                f"{algo_name} CUDA inference transport requires rank-local devices to "
                f"match; got env={env_resolved!r}, learner={learner!r}"
            )
        return InferencePlacement(
            mode=InferenceTransport.CUDA,
            env_device=env_resolved,
            ring_device=learner,
            learner_device=learner,
            collector_tensor_native=True,
            staging_policy="cuda_no_host_boundary",
        )

    if _device_requires_cuda(env_resolved):
        raise ValueError(
            f"{algo_name} host inference transport requires a CPU env device, "
            f"got env={env_resolved!r}"
        )
    return InferencePlacement(
        mode=InferenceTransport.HOST,
        env_device=env_resolved,
        ring_device="cpu",
        learner_device=learner,
        collector_tensor_native=False,
        staging_policy=cpu_ring_staging_policy(learner),
    )
