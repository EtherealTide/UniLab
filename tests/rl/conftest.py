"""Shared fixtures for unilab.rl tests.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15) tests/conftest.py. The
fake env classes live in :mod:`tests.rl._fakes` so spawn subprocesses can
import them without executing this conftest.
"""

from __future__ import annotations

import pytest
import torch

from tests.rl._fakes import FakeEnvFactory


@pytest.fixture
def tiny_weight_shapes():
    """Small MLP param shapes dict — linear(8,16) + bias, linear(16,3) + bias."""
    return {
        "layer1.weight": torch.Size([16, 8]),
        "layer1.bias": torch.Size([16]),
        "layer2.weight": torch.Size([3, 16]),
        "layer2.bias": torch.Size([3]),
    }


@pytest.fixture
def fake_env_factory() -> FakeEnvFactory:
    return FakeEnvFactory()
