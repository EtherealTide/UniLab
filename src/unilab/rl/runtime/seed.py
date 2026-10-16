"""Worker seed helpers for RL runtime processes.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.seed``. The owner-config seed resolution contract
(``resolve_training_seed`` / ``TrainingSeedInfo``) is owned by
:mod:`unilab.training.seed`; this module holds the runtime-side seed
application and per-worker derivation.
"""

from __future__ import annotations

import random

import numpy as np


def derive_worker_seed(base_seed: int | None, worker_index: int = 0) -> int | None:
    """Derive deterministic subprocess seeds from the effective run seed."""
    if base_seed is None:
        return None
    if worker_index < 0:
        raise ValueError(f"worker_index must be non-negative, got {worker_index}")
    return int(base_seed) + int(worker_index) + 1


def apply_training_seed(
    seed: int | None,
    *,
    torch_runtime: bool = True,
    cuda: bool = True,
) -> int | None:
    """Apply a seed to the runtimes used by training entrypoints."""
    if seed is None:
        return None

    effective_seed = int(seed)
    if effective_seed < 0:
        raise ValueError(f"seed must be non-negative, got {effective_seed}")

    random.seed(effective_seed)
    np.random.seed(effective_seed)

    if torch_runtime:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.manual_seed(effective_seed)
            if cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(effective_seed)

    return effective_seed
