"""Replay transfer backend factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from unilab.rl.runtime.ipc.replay_pipelines.transfer.base import ReplayTransferBackend
from unilab.rl.runtime.ipc.replay_pipelines.transfer.cuda_like import CudaLikeReplayTransferBackend
from unilab.rl.runtime.ipc.replay_pipelines.transfer.torch_copy import (
    TorchCopyReplayTransferBackend,
)


def build_replay_transfer_backend(
    *,
    device: torch.device,
    ring_depth: int,
) -> ReplayTransferBackend:
    """Build the transfer backend for a learner device."""
    builders = {
        "cuda": CudaLikeReplayTransferBackend,
        "mps": TorchCopyReplayTransferBackend,
    }
    try:
        backend_type = cast(Callable[..., ReplayTransferBackend], builders[device.type])
    except KeyError as exc:
        raise ValueError(
            f"Replay transfer requires a CUDA or MPS device, got {device.type!r}"
        ) from exc
    return backend_type(device=device, ring_depth=ring_depth)
