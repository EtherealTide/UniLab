"""RSL-RL distributed-training helpers.

The small pure helpers here (torchrun environment readers, removed-device
configuration validation) are owned by UniLab so that single-process PPO
training and playback run without uni_rl installed. The multi-GPU launcher and
CPU partitioner stay owned by ``uni_rl.ipc.dp_launcher`` and are imported
lazily only when a multi-rank topology is requested.
``UNILAB_DP_LOG_DIR`` is a shared environment-variable contract with that
launcher: it sets the variable for spawned workers, workers read it here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from omegaconf import open_dict

UNILAB_DP_LOG_DIR = "UNILAB_DP_LOG_DIR"


def current_torch_distributed_rank() -> int:
    """Global torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("RANK", "0"))


def current_torch_distributed_local_rank() -> int:
    """Node-local torch-distributed rank of this process (0 outside torchrun)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def current_torch_distributed_world_size() -> int:
    """Torch-distributed world size of this process (1 outside torchrun)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def reject_removed_device_config(devices_cfg: Any) -> None:
    """Fail closed if the removed CUDA-index device list is still supplied."""
    if devices_cfg is None:
        return
    if hasattr(devices_cfg, "__len__") and len(devices_cfg) == 0:
        return
    raise ValueError(
        "training.devices was removed; control GPUs exclusively with rank-local "
        "CUDA_VISIBLE_DEVICES"
    )


def _require_uni_rl_dp_launcher():
    """Import uni_rl's data-parallel launcher or raise an actionable error."""
    try:
        from uni_rl.ipc import dp_launcher
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] != "uni_rl":
            raise
        raise ModuleNotFoundError(
            "Multi-GPU PPO and multi-rank CPU partitioning require the optional "
            "unilab-rl package; "
            "install it with: pip install unilab[uni_rl]"
        ) from exc
    return dp_launcher


def resolve_collector_cpu_ids(
    world_size: int,
    rank: int,
    cpu_count: int | None = None,
    explicit: Any = None,
) -> list[int] | None:
    """Resolve the CPU ids exclusively owned by this rank's collector.

    Single-rank runs return None without touching uni_rl; multi-rank CPU
    partitioning is owned by ``uni_rl.ipc.dp_launcher`` and requires the
    optional unilab-rl package.
    """
    if int(world_size) <= 1:
        return None
    result = _require_uni_rl_dp_launcher().resolve_collector_cpu_ids(
        world_size,
        rank,
        cpu_count,
        explicit=explicit,
    )
    return cast("list[int] | None", result)


def launch_torchrun_workers(
    *,
    world_size: int,
    script_path: str | os.PathLike[str],
    argv: Sequence[str],
    log_dir: str,
) -> None:
    """Launch one single-GPU torchrun worker per selected visibility entry.

    Worker supervision is owned by ``uni_rl.ipc.dp_launcher`` and requires the
    optional unilab-rl package.
    """
    _require_uni_rl_dp_launcher().launch_torchrun_workers(
        world_size=world_size,
        script_path=Path(script_path),
        argv=argv,
        log_dir=log_dir,
    )


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
    from unilab.base.process_device import rank_local_visible_cuda_entries

    entries = rank_local_visible_cuda_entries()
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
