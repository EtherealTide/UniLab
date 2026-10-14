"""Focused real-CUDA tests for the state-update-only graph runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnv

pytestmark = pytest.mark.slow
ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"


def _require_cuda_mjwarp() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("env update graph CUDA tests require an active CUDA Warp device")


def _materialize_env() -> Any:
    _require_cuda_mjwarp()
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        hydra_cfg = compose("config", overrides=["task=g1_walk_flat/mjwarp"])
    override = BackendAdapter(hydra_cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg = cast(Any, registry.materialize_env_config("G1WalkFlat"))
    apply_cfg_overrides(env_cfg, override)
    env_cfg.validate()
    assert env_cfg.scene is not None
    env_cfg.scene.default_keyframe_name = "stand"
    env_cfg.commands = {}
    env_cfg.observations["policy"].enable_corruption = False
    env_cfg.observations["critic"].enable_corruption = False
    from unilab.managers import ObservationTermCfg

    def policy_obs(_env):
        return torch.zeros((2, 1), dtype=torch.float32, device="cuda")

    def critic_obs(_env):
        return torch.zeros((2, 2), dtype=torch.float32, device="cuda")

    setattr(policy_obs, "returns_transient_tensor", True)
    setattr(critic_obs, "returns_transient_tensor", True)
    env_cfg.observations["policy"].terms = {"constant": ObservationTermCfg(func=policy_obs)}
    env_cfg.observations["critic"].terms = {"constant": ObservationTermCfg(func=critic_obs)}
    for group in env_cfg.observations.values():
        for term in group.terms.values():
            term.noise = None
    # Start with an allocation-safe scalar reward and add graph terms after the
    # first production candidate proves the fixed carrier invariant.
    from unilab.managers import RewardTermCfg

    def reward(env):
        return env.action_manager.action[:, 0]

    setattr(reward, "returns_transient_tensor", True)
    env_cfg.rewards = {"constant": RewardTermCfg(func=reward, weight=1.0)}
    env_cfg.terminations = {
        "time_out": env_cfg.terminations["time_out"],
    }
    # Revalidate after the focused first-candidate terms remove configured noise.
    assert env_cfg.max_episode_seconds == 20.0
    env_cfg.validate()
    del override
    from unilab.tasks.locomotion.g1.manager_terms import make_g1_walk_env

    return make_g1_walk_env(env_cfg, num_envs=2, backend_type="mjwarp")


def test_real_cuda_env_update_graph_capture_replay_and_allocator_invariant() -> None:
    env = _materialize_env()
    try:
        assert isinstance(env, ManagerBasedRlEnv)
        assert env.device.type == "cuda"
        env.set_deferred_validation_boundaries(True)
        env.reset(seed=7)
        env.configure_env_update_graph("graph")

        manifest = env.get_env_update_graph_manifest()
        assert manifest["enabled"] is True
        assert manifest["state"] == "ready"
        assert manifest["replay_count"] == 1

        actions = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
        for _ in range(4):
            env.step(actions)
            manifest = env.get_env_update_graph_manifest()
            assert manifest["enabled"] is True
            assert manifest["state"] == "ready"
            assert manifest["replay_count"] > 1
    finally:
        env.close()
