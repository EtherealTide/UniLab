"""Hydra-side data-parallel configuration assembly.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.ipc.dp_launcher`` (and duplicated in ``unilab.rl.distributed``). This
module owns the per-rank Hydra config rewrite; rank/device topology lives in
:mod:`unilab.rl.runtime.distributed.context`.
"""

from __future__ import annotations

from typing import Any

from unilab.rl.runtime.distributed.context import rank_local_cuda_device


def apply_dp_rank_config(cfg: Any, rank: int) -> str | None:
    """Apply the per-rank seed and return this rank's visibility-local device.

    Rank 0 keeps the configured seed; rank i>0 trains with ``seed + i`` until
    init broadcast lands in a later stage. The resolved rank-local device is
    returned instead of being written back into a synthetic ``training.device``
    key; with rank-local CUDA visibility this is always ``cuda:0``.
    """
    device = rank_local_cuda_device(rank=rank)
    if device is None:
        return None
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.algo.seed = int(cfg.algo.seed) + rank
    return device
