"""Hydra-side data-parallel configuration assembly.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.ipc.dp_launcher`` (and duplicated in ``unilab.rl.distributed``). This
module is the single owner of the removed-device configuration guard and the
per-rank Hydra config rewrite; rank/device topology itself lives in
:mod:`unilab.rl.runtime.distributed.context`.
"""

from __future__ import annotations

from typing import Any

from unilab.rl.runtime.distributed.context import resolve_dp_rank_device


def reject_removed_device_config(devices_cfg: Any) -> None:
    """Fail closed if a removed CUDA-index device list is still supplied.

    GPU topology is controlled exclusively by rank-local
    ``CUDA_VISIBLE_DEVICES``. A CUDA-index list in configuration would create a
    second topology source and cannot be mapped safely when parent visibility
    contains opaque UUID/MIG tokens.
    """
    if devices_cfg is None:
        return
    if hasattr(devices_cfg, "__len__") and len(devices_cfg) == 0:
        return
    raise ValueError(
        "training.devices was removed; control GPUs exclusively with rank-local "
        "CUDA_VISIBLE_DEVICES"
    )


def apply_dp_rank_config(cfg: Any, rank: int) -> str | None:
    """Apply the per-rank seed and return this rank's visibility-local device.

    Rank 0 keeps the configured seed; rank i>0 trains with ``seed + i`` until
    init broadcast lands in a later stage. The resolved rank-local device is
    returned instead of being written back into a synthetic ``training.device``
    key; with rank-local CUDA visibility this is always ``cuda:0``.
    """
    device = resolve_dp_rank_device(rank)
    if device is None:
        return None
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.algo.seed = int(cfg.algo.seed) + rank
    return device
