"""Test reward config override through registry."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.registry import ensure_registries

ROOT_DIR = Path(__file__).parents[2]


def test_g1_reset_randomization_fails_closed_for_reward_override_fixture():
    """Reward overrides cannot bypass the fixed-width reset contract."""
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    ensure_registries()

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "src" / "unilab" / "conf" / "ppo"), version_base="1.3"
    ):
        cfg = compose("config", overrides=["task=g1_walk_flat/mujoco"])
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg_override["rewards"]["tracking_lin_vel"]["weight"] = 888.0
    env_cfg_override["rewards"]["alive"] = {
        "_target_": "unilab.managers.RewardTermCfg",
        "func": "unilab.tasks.locomotion.common.manager_terms.alive",
        "weight": 20.0,
    }

    with pytest.raises(NotImplementedError, match="reset randomization"):
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override=env_cfg_override,
        )
