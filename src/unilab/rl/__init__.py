"""UniLab RL subsystem: contracts, adapters, algorithms, runtime, logging.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0.

This package init is intentionally lightweight: importing ``unilab.rl`` must
not pull in Torch, TensorDict, or any algorithm stack. The public surface
below is resolved lazily via PEP 562 ``__getattr__`` so existing callers keep
their ``from unilab.rl import ...`` entry points while the implementations
live in their owner modules.
"""

from __future__ import annotations

from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "UNILAB_DP_LOG_DIR": ("unilab.rl.runtime.distributed.context", "UNILAB_DP_LOG_DIR"),
    "RslRlVecEnvAdapter": ("unilab.rl.adapters.rsl_rl.vec_env", "RslRlVecEnvAdapter"),
    "apply_rsl_rl_rank_seed": ("unilab.rl.adapters.rsl_rl.distributed", "apply_rsl_rl_rank_seed"),
    "current_torch_distributed_local_rank": (
        "unilab.rl.runtime.distributed.context",
        "current_torch_distributed_local_rank",
    ),
    "current_torch_distributed_rank": (
        "unilab.rl.runtime.distributed.context",
        "current_torch_distributed_rank",
    ),
    "current_torch_distributed_world_size": (
        "unilab.rl.runtime.distributed.context",
        "current_torch_distributed_world_size",
    ),
    "finish_rsl_rl_distributed": (
        "unilab.rl.adapters.rsl_rl.distributed",
        "finish_rsl_rl_distributed",
    ),
    "get_policy_obs_dims": ("unilab.rl.adapters.rsl_rl.vec_env", "get_policy_obs_dims"),
    "launch_torchrun_workers": (
        "unilab.rl.runtime.distributed.launcher",
        "launch_torchrun_workers",
    ),
    "ppo_samples_per_iteration": (
        "unilab.rl.adapters.rsl_rl.distributed",
        "ppo_samples_per_iteration",
    ),
    "reject_removed_device_config": ("unilab.training.dp_config", "reject_removed_device_config"),
    "resolve_collector_cpu_ids": (
        "unilab.rl.runtime.distributed.affinity",
        "resolve_collector_cpu_ids",
    ),
    "resolve_rsl_rl_device": ("unilab.rl.adapters.rsl_rl.distributed", "resolve_rsl_rl_device"),
    "rsl_rl_single_process_topology": (
        "unilab.rl.adapters.rsl_rl.distributed",
        "rsl_rl_single_process_topology",
    ),
}

__all__ = [
    "RslRlVecEnvAdapter",
    "UNILAB_DP_LOG_DIR",
    "apply_rsl_rl_rank_seed",
    "current_torch_distributed_local_rank",
    "current_torch_distributed_rank",
    "current_torch_distributed_world_size",
    "finish_rsl_rl_distributed",
    "get_policy_obs_dims",
    "launch_torchrun_workers",
    "ppo_samples_per_iteration",
    "reject_removed_device_config",
    "resolve_collector_cpu_ids",
    "resolve_rsl_rl_device",
    "rsl_rl_single_process_topology",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = target
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
