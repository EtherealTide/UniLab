"""RSL-RL (PPO) distributed-training helpers.

Torchrun environment readers and the ``UNILAB_DP_LOG_DIR`` contract are owned
by :mod:`unilab.rl.runtime.distributed.context` and imported here; this module
only holds the PPO-specific rank seed, device, sample-count, shutdown, and
playback-topology helpers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from omegaconf import open_dict

from unilab.process_context import visible_cuda_entries
from unilab.rl.runtime.distributed.context import (
    UNILAB_DP_LOG_DIR,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
)

__all__ = [
    "UNILAB_DP_LOG_DIR",
    "apply_rsl_rl_rank_seed",
    "current_torch_distributed_local_rank",
    "current_torch_distributed_rank",
    "current_torch_distributed_world_size",
    "finish_rsl_rl_distributed",
    "ppo_samples_per_iteration",
    "resolve_rsl_rl_device",
    "rsl_rl_single_process_topology",
]


def apply_rsl_rl_rank_seed(cfg: Any, rank: int) -> int:
    """Apply RSL-RL's ``base seed + global rank`` data-parallel contract."""
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    base_seed = int(cfg.algo.seed)
    with open_dict(cfg):
        cfg.algo.seed = base_seed + int(rank)
    return int(cfg.algo.seed)


def resolve_rsl_rl_device(
    *,
    configured_device: str | None,
    world_size: int,
    local_rank: int,
    default_device: str,
) -> str:
    """Resolve the exact device string expected by RSL-RL's runner.

    Every distributed worker owns one visible GPU and therefore uses
    ``cuda:0``. ``LOCAL_RANK`` is a process label, never a CUDA index.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if local_rank < 0 or local_rank >= world_size:
        raise ValueError(f"local_rank={local_rank} is out of range for world_size={world_size}")
    entries = visible_cuda_entries()
    if len(entries) == 1:
        return "cuda:0"
    if entries:
        raise ValueError(
            "A CUDA PPO worker must own exactly one CUDA_VISIBLE_DEVICES entry; "
            f"got {','.join(entries)!r}"
        )
    return configured_device or default_device


def ppo_samples_per_iteration(*, num_envs: int, num_steps_per_env: int, world_size: int) -> int:
    """Return the global fresh rollout sample count for one PPO iteration."""
    return int(num_envs) * int(num_steps_per_env) * int(world_size)


def finish_rsl_rl_distributed(*, training_succeeded: bool) -> None:
    """Synchronize successful ranks and release RSL-RL's process group."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    try:
        if training_succeeded:
            torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@contextmanager
def rsl_rl_single_process_topology() -> Iterator[None]:
    """Temporarily hide torchrun's worker topology from rank-0-only work.

    Destroying a process group does not clear ``WORLD_SIZE`` / ``RANK`` /
    ``LOCAL_RANK``. RSL-RL would therefore initialize a second distributed
    group when rank 0 constructs a fresh runner for post-training playback,
    even though every other rank has already exited. Present the playback
    scope as a single-process runtime, then restore launcher-owned variables.
    """
    single_process_topology = {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
    }
    previous = {name: os.environ.get(name) for name in single_process_topology}
    os.environ.update(single_process_topology)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
