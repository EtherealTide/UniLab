"""Stdlib-only process visibility context shared by UniLab layers.

This module owns the process-level environment contracts shared across layers:
``CUDA_VISIBLE_DEVICES`` token parsing for ``unilab.base`` (backend device
routing) and ``unilab.rl.runtime`` (data-parallel launch topology), plus the
debug opt-in for the CUDA-learner CPU-carrier prohibition (issue #2483). It
must stay stdlib-only: it is usable before the first CUDA query and must not
import Torch.
"""

from __future__ import annotations

import os

CUDA_LEARNER_CPU_CARRIER_OPT_IN = "UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER"

_OPT_IN_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def cuda_learner_cpu_carrier_opted_in(current_value: str | None = None) -> bool:
    """Return whether the debug opt-in allows a CPU carrier with a CUDA learner.

    Issue #2483 makes ``collector_tensor_device=cpu`` with a CUDA learner fail
    closed: the CPU carrier is retained only for CPU/torch.mps learners,
    ROCm/HIP builds, and explicit attribution-baseline or debug runs. Those
    runs set ``UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER`` to a true value
    (``1``/``true``/``yes``/``on``); anything else keeps the prohibition.
    """

    raw = (
        os.environ.get(CUDA_LEARNER_CPU_CARRIER_OPT_IN) if current_value is None else current_value
    )
    return raw is not None and raw.strip().lower() in _OPT_IN_TRUE_VALUES


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
