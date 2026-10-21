"""Data-parallel rank topology context (stdlib-only, Torch lazily loaded).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.ipc.dp_launcher``. ``CUDA_VISIBLE_DEVICES`` token parsing is delegated
to :mod:`unilab.process_context`, the single owner of that contract.
"""

from __future__ import annotations

import os

from unilab.process_context import visible_cuda_entries

UNILAB_DP_RANK = "UNILAB_DP_RANK"
UNILAB_DP_WORLD_SIZE = "UNILAB_DP_WORLD_SIZE"
UNILAB_DP_LOG_DIR = "UNILAB_DP_LOG_DIR"


def current_dp_rank() -> int:
    """Data-parallel rank of this process (0 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_RANK, "0"))


def rank_local_cuda_device(
    *,
    rank: int = 0,
    current_visible_devices: str | None = None,
) -> str | None:
    """Resolve ``cuda:0`` when this rank owns exactly one visible GPU.

    ``CUDA_VISIBLE_DEVICES`` is the authoritative rank-local namespace.  A
    single entry can come either from an ordinary user launch or from
    :class:`~unilab.rl.runtime.distributed.launcher.DpRankSupervisor`; in both
    cases every in-process consumer must use the local index zero rather than a
    host-global index.

    On a host with exactly one visible CUDA device, leaving the variable unset
    is equivalent to a one-entry mask: the rank owns local ordinal zero. This
    honors the documented single-GPU launch contract without guessing on a
    multi-GPU host.
    """
    entries = visible_cuda_entries(current_visible_devices)
    if entries:
        if len(entries) != 1:
            raise ValueError(
                "A CUDA rank must own exactly one CUDA_VISIBLE_DEVICES entry; got "
                f"{','.join(entries)!r}"
            )
    else:
        import torch

        try:
            visible_to_torch = torch.cuda.is_available() and torch.cuda.device_count() == 1
        except RuntimeError:
            visible_to_torch = False
        if not visible_to_torch:
            return None
    if int(rank) < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    return "cuda:0"


def current_dp_world_size() -> int:
    """Data-parallel world size of this process (1 when not spawned as a rank)."""
    return int(os.environ.get(UNILAB_DP_WORLD_SIZE, "1"))


def current_torch_distributed_rank() -> int:
    """Global torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("RANK", "0"))


def current_torch_distributed_local_rank() -> int:
    """Node-local torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def current_torch_distributed_world_size() -> int:
    """Torch-distributed world size of this process (1 outside torchrun)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def resolve_dp_rendezvous_path(log_dir: str, *, rank: int) -> str:
    """Shared FileStore rendezvous path for the DP parameter-sync group.

    Rank 0 anchors the store in its own run directory; spawned ranks point at
    the same canonical run root via ``UNILAB_DP_LOG_DIR``. The run directory
    is unique per run, which keeps stale FileStore state from a previous run
    out of the rendezvous.

    Cold path only: call at runner construction.
    """
    if int(rank) > 0:
        root = os.environ.get(UNILAB_DP_LOG_DIR)
        if root is None:
            raise ValueError(
                f"{UNILAB_DP_LOG_DIR} must be set for spawned data-parallel rank {rank}"
            )
    else:
        root = str(log_dir)
    return os.path.join(root, ".dp_rendezvous")


def selected_visible_entries(
    *,
    world_size: int,
    current_visible_devices: str | None = None,
) -> tuple[str, ...]:
    """Return the opaque parent visibility entries selected for a DP launch.

    Parent ``CUDA_VISIBLE_DEVICES`` may contain physical indices, UUIDs, or MIG
    tokens. The first ``world_size`` entries define rank order; no integer
    configuration index is involved.
    """
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError("multi-rank visibility requires world_size >= 1")
    entries = visible_cuda_entries(current_visible_devices)
    if len(entries) != world_size:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one entry per data-parallel "
            f"rank for this launch; got {len(entries)} entries for world_size={world_size}"
        )
    if len(set(entries)) != len(entries):
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains duplicate entries: {entries}")
    return entries
