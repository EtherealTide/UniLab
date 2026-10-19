from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.training.seed import (
    apply_training_seed,
    resolve_training_seed,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_CONF_DIR = _ROOT_DIR / "src" / "unilab" / "conf"


def _compose(config_dir: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / config_dir), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def test_resolve_training_seed_uses_algo_seed_and_ignores_training_seed():
    cfg = OmegaConf.create({"algo": {"seed": 7}, "training": {"seed": 99}})

    seed_info = resolve_training_seed(cfg)

    assert seed_info.configured_seed == 7
    assert seed_info.configured_seed_source == "algo.seed"
    assert seed_info.effective_seed == 7


def test_resolve_training_seed_fails_closed_on_training_seed_without_algo_seed():
    cfg = OmegaConf.create({"training": {"seed": 99}})

    with pytest.raises(ValueError, match="algo.seed"):
        resolve_training_seed(cfg)


def test_resolve_training_seed_returns_empty_metadata_without_configured_seed():
    seed_info = resolve_training_seed(OmegaConf.create({}))

    assert seed_info.configured_seed is None
    assert seed_info.configured_seed_source is None
    assert seed_info.effective_seed is None


def test_resolve_training_seed_rejects_negative_algo_seed():
    cfg = OmegaConf.create({"algo": {"seed": -1}})

    with pytest.raises(ValueError, match="algo.seed must be non-negative"):
        resolve_training_seed(cfg)


def test_apply_training_seed_controls_python_and_torch_rng_without_numpy():
    before = np.random.get_state()
    apply_training_seed(123, torch_runtime=True, cuda=True)
    first = (random.random(), torch.rand(3))
    after_seed = np.random.get_state()

    apply_training_seed(123, torch_runtime=True, cuda=True)
    second = (random.random(), torch.rand(3))

    assert second[0] == first[0]
    assert torch.equal(second[1], first[1])
    assert before[0] == after_seed[0]
    np.testing.assert_array_equal(before[1], after_seed[1])
    assert before[2:] == after_seed[2:]


def test_apply_training_seed_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        apply_training_seed(-1)


@pytest.mark.parametrize(
    ("config_dir", "overrides"),
    [
        ("ppo", ["task=go2_joystick_flat/mujoco"]),
        ("ppo", ["task=go2_joystick_flat/mujoco", "algo.seed=41"]),
        ("appo", ["task=go2_joystick_flat/mujoco"]),
        ("sac", ["task=g1_walk_flat/mujoco"]),
    ],
)
def test_owner_configs_resolve_algorithm_seed_contract(config_dir: str, overrides: list[str]):
    cfg = _compose(config_dir, overrides)

    seed_info = resolve_training_seed(cfg)

    assert seed_info.configured_seed == int(cfg.algo.seed)
    assert seed_info.configured_seed_source == "algo.seed"
    assert seed_info.effective_seed == int(cfg.algo.seed)
