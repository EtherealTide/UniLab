"""Stdlib-only process visibility context shared by UniLab layers.

This module is the single owner of ``CUDA_VISIBLE_DEVICES`` token parsing so
``unilab.base`` (backend device routing) and ``unilab.rl.runtime`` (data-parallel
launch topology) share one implementation. It must stay stdlib-only: it is
usable before the first CUDA query and must not import Torch.
"""

from __future__ import annotations

import os


def visible_cuda_entries(current_visible_devices: str | None = None) -> tuple[str, ...]:
    """Return the process-local CUDA visibility entries without touching Torch.

    Entries are opaque: they may be physical indices or CUDA UUID/MIG tokens.
    An unset variable means the host namespace; an empty value (or ``-1``)
    means CUDA is deliberately hidden. This parser is cold-path only and is
    intentionally usable before the first CUDA query.
    """
    raw = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if current_visible_devices is None
        else current_visible_devices
    )
    if raw is None:
        return ()
    entries = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    if entries == ("-1",):
        return ()
    return entries
