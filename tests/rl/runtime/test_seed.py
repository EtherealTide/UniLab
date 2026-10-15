"""Runtime seed helpers (owner: rl.runtime.seed).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/utils/test_seed_contract.py``. Owner-config seed resolution
(``resolve_training_seed`` / ``TrainingSeedInfo``) is owned by
:mod:`unilab.training.seed` and covered by
``tests/training/test_seed_contract.py``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from unilab.rl.runtime.seed import apply_training_seed, derive_worker_seed


def test_apply_training_seed_controls_python_numpy_and_torch_rng():
    apply_training_seed(123, torch_runtime=True, cuda=True)
    first = (random.random(), np.random.rand(), torch.rand(3))

    apply_training_seed(123, torch_runtime=True, cuda=True)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert second[0] == first[0]
    assert second[1] == first[1]
    assert torch.equal(second[2], first[2])


def test_apply_training_seed_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        apply_training_seed(-1)


def test_derive_worker_seed_is_deterministic_and_distinct_from_base_seed():
    assert derive_worker_seed(10, worker_index=0) == 11
    assert derive_worker_seed(10, worker_index=3) == 14
    assert derive_worker_seed(None, worker_index=3) is None
