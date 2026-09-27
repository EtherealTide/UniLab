from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1 import torch_flashsac_env as module

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"


def _materialize_task(task: str) -> ManagerBasedRlEnvCfg:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        composed = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(
        composed, root_dir=ROOT_DIR, algo_name="flashsac"
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def test_torch_owner_fingerprint_accepts_both_canonical_backends() -> None:
    mujoco = _materialize_task("g1_motion_tracking/mujoco")
    mjwarp = _materialize_task("g1_motion_tracking/mjwarp")
    expected = module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1
    assert module._torch_g1_flashsac_owner_identity(mujoco) == expected
    assert module._torch_g1_flashsac_owner_identity(mjwarp) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.observations["actor"].terms["base_lin_vel"].noise, "n_min", -0.2),
        lambda cfg: cfg.rewards["motion_global_root_pos"].params.__setitem__("std", 0.4),
        lambda cfg: setattr(cfg.actions["joint_pos"], "clip", {"joint": (0.0, 1.0)}),
        lambda cfg: setattr(cfg.commands["motion"].params, "adaptive_alpha", 0.1),
    ],
)
def test_torch_owner_fingerprint_fails_closed(mutate, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    mutate(cfg)

    def fail(_name, *_args, **_kwargs):
        raise AssertionError("backend must not be created for a mutated owner")

    monkeypatch.setattr(module, "create_backend", fail)
    with pytest.raises(ValueError, match="canonical owner contract"):
        module.make_torch_g1_motion_tracking_flashsac_env(cfg, num_envs=2, backend_type="mujoco")


def test_torch_owner_rejects_fixed_model_variants_before_backend_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    cfg.fixed_model_variants = FixedModelVariantCatalogCfg(
        variants=(FixedModelVariantCfg(name="variant", source_model_file="variant.xml"),)
    )

    def fail(_name, *_args, **_kwargs):
        raise AssertionError("fixed variants must fail before backend creation")

    monkeypatch.setattr(module, "create_backend", fail)
    with pytest.raises(ValueError, match="does not support fixed model variants"):
        module.make_torch_g1_motion_tracking_flashsac_env(cfg, num_envs=2, backend_type="mujoco")


def test_torch_observation_noise_uses_configured_bounds_and_keeps_critic_clean() -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    num_envs, num_joints, num_bodies = 2, 29, 2
    device = torch.device("cpu")
    env._torch = torch  # pyright: ignore[reportAttributeAccessIssue]
    env.device = device
    env._motion_joint_pos = torch.arange(num_envs * num_joints, dtype=torch.float32).reshape(
        num_envs, num_joints
    )
    env._motion_joint_vel = torch.full((num_envs, num_joints), 0.25)
    env._motion_anchor_pos_b = torch.full((num_envs, 3), 0.5)
    env._motion_anchor_ori_b = torch.full((num_envs, 6), -0.5)
    env._linvel = torch.full((num_envs, 3), 1.0)
    env._gyro = torch.full((num_envs, 3), -1.0)
    env._default_joint_pos = torch.full((num_envs, num_joints), 0.125)
    env._joint_default_bias = torch.full((num_envs, num_joints), -0.25)
    env._default_joint_vel = torch.full((num_envs, num_joints), 0.5)
    env._joint_pos = torch.full((num_envs, num_joints), 0.75)
    env._joint_vel = torch.full((num_envs, num_joints), -0.75)
    env._raw_actions = torch.linspace(-1, 1, num_envs * num_joints).reshape(num_envs, num_joints)
    env._robot_body_pos_b = torch.full((num_envs, num_bodies, 3), 2.0)
    env._robot_body_ori_b = torch.full((num_envs, num_bodies, 6), -2.0)
    env._actor_corruption = True
    env._observation_noise_lower = torch.tensor([-0.1, -0.2, -0.01, -1.5] * 16, dtype=torch.float32)
    env._observation_noise_upper = torch.tensor([0.1, 0.2, 0.01, 1.5] * 16, dtype=torch.float32)
    env._rng = torch.Generator(device=device).manual_seed(7)

    result = env._compute_observations()

    actor = result["obs"]
    critic = result["critic"]
    cursor = num_joints * 2 + 3 + 6
    width = env._observation_noise_lower.numel()
    noise_generator = torch.Generator(device=device).manual_seed(7)
    noise = torch.rand((num_envs, width), generator=noise_generator, device=device)
    noise *= env._observation_noise_upper - env._observation_noise_lower
    noise += env._observation_noise_lower
    torch.testing.assert_close(
        actor[:, cursor : cursor + width], critic[:, cursor : cursor + width] + noise
    )
    torch.testing.assert_close(actor[:, :cursor], critic[:, :cursor])


def test_manual_torch_reset_clears_only_selected_done_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._num_envs = 3
    env.device = torch.device("cpu")
    env._state = module.TorchEnvState(
        obs={"obs": torch.zeros((3, 2))},
        reward=torch.zeros(3),
        terminated=torch.tensor([True, False, False]),
        truncated=torch.tensor([False, True, False]),
        info={},
    )
    monkeypatch.setattr(env, "_reset_rows", lambda rows, obs: obs)
    obs, info = env.reset(env_indices=torch.tensor([0, 2]))
    assert obs["obs"].shape == (2, 2)
    assert info == {"log": {}}
    assert not bool(env._state.terminated.any())
    assert torch.equal(env._state.truncated, torch.tensor([False, True, False]))


def test_torch_mjwarp_renegotiates_refresh_and_preserves_policy_sensor_boundary() -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._torch = torch  # pyright: ignore[reportAttributeAccessIssue]
    env.device = torch.device("cpu")
    env._body_names = ("pelvis", "torso")
    env._joint_qpos_ids = np.array([7, 8], dtype=np.int64)
    env._joint_qvel_ids = np.array([6, 7], dtype=np.int64)
    env._qpos = torch.zeros((2, 9))
    env._qvel = torch.zeros((2, 8))
    env._linvel = torch.ones((2, 3))
    env._gyro = torch.full((2, 3), -1.0)
    linvel_view = env._linvel.clone()
    gyro_view = env._gyro.clone()
    position_views = tuple(torch.ones((2, 3)) for _ in range(2))
    quat_views = tuple(torch.ones((2, 4)) for _ in range(2))
    env._mjwarp_state_views = {"qpos": env._qpos, "qvel": env._qvel}
    env._mjwarp_linvel_view = linvel_view
    env._mjwarp_gyro_view = gyro_view
    env._mjwarp_sensor_views = {
        "linvel": linvel_view,
        "gyro": gyro_view,
        "track_pos_w": position_views,
        "track_quat_w": quat_views,
        "track_linvel_w": position_views,
        "track_angvel_w": position_views,
    }
    env._robot_body_pos = torch.zeros((2, 2, 3))
    env._robot_body_quat = torch.zeros((2, 2, 4))
    env._robot_body_lin_vel = torch.zeros((2, 2, 3))
    env._robot_body_ang_vel = torch.zeros((2, 2, 3))

    def refresh_first_tracking_sensor(name: str, device=None):
        assert name == "track_pos_w_pelvis"
        assert device == env.device
        for view in position_views:
            view.fill_(2.0)
        return position_views[0]

    env._backend = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        backend_type="mjwarp", get_sensor_view=refresh_first_tracking_sensor
    )
    env._read_robot_state()

    torch.testing.assert_close(env._linvel, torch.ones((2, 3)))
    torch.testing.assert_close(env._gyro, torch.full((2, 3), -1.0))
    torch.testing.assert_close(env._robot_body_pos, torch.full((2, 2, 3), 2.0))
