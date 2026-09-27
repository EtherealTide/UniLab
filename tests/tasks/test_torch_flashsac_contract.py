from __future__ import annotations

from pathlib import Path

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


def _materialize_task(task: str, *, algo: str = "flashsac") -> ManagerBasedRlEnvCfg:
    GlobalHydra.instance().clear()
    config_dir = CONF_DIR if algo == "flashsac" else ROOT_DIR / "src/unilab/conf/sac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(
        composed, root_dir=ROOT_DIR, algo_name=algo
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


def test_reusable_tensor_runtime_accepts_second_g1_manager_owner() -> None:
    cfg = _materialize_task("g1_motion_tracking/mjwarp_tensor", algo="sac")
    assert cfg.tensor_runtime is True
    assert module._torch_g1_flashsac_owner_identity(cfg) == module._TORCH_G1_SAC_OWNER_IDENTITY_V1


def test_reusable_tensor_runtime_accepts_second_manager_based_task() -> None:
    cfg = _materialize_task("g1_flip_tracking/mjwarp_tensor", algo="sac")
    assert cfg.tensor_runtime is True
    assert cfg.commands["motion"].sampling_mode == "mixed"
    assert module._torch_g1_flashsac_owner_identity(cfg) == (
        module._TORCH_G1_FLIP_SAC_OWNER_IDENTITY_V1
    )


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
    env._actor_joint_pos_biased = False
    env._critic_prefix_names = (
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    env._observation_noise = module.TensorObservationNoise(
        torch.tensor([-0.1, -0.2, -0.01, -1.5] * 16, dtype=torch.float32),
        torch.tensor([0.1, 0.2, 0.01, 1.5] * 16, dtype=torch.float32),
    )
    env._rng = torch.Generator(device=device).manual_seed(7)

    result = env._compute_observations()

    actor = result["obs"]
    critic = result["critic"]
    cursor = num_joints * 2 + 3 + 6
    width = env._observation_noise.lower.numel()
    noise_generator = torch.Generator(device=device).manual_seed(7)
    noise = torch.rand((num_envs, width), generator=noise_generator, device=device)
    noise *= env._observation_noise.upper - env._observation_noise.lower
    noise += env._observation_noise.lower
    torch.testing.assert_close(
        actor[:, cursor : cursor + width], critic[:, cursor : cursor + width] + noise
    )
    torch.testing.assert_close(actor[:, :cursor], critic[:, :cursor])


def test_second_g1_owner_keeps_actor_encoder_bias_out_of_critic() -> None:
    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env._torch = torch
    env.device = torch.device("cpu")
    env._actor_joint_pos_biased = True
    env._actor_corruption = False
    env._critic_prefix_names = (
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "joint_vel",
        "actions",
        "joint_pos",
    )
    env._motion_joint_pos = torch.zeros((2, 29))
    env._motion_joint_vel = torch.zeros((2, 29))
    env._motion_anchor_pos_b = torch.zeros((2, 3))
    env._motion_anchor_ori_b = torch.zeros((2, 6))
    env._linvel = torch.zeros((2, 3))
    env._gyro = torch.zeros((2, 3))
    env._joint_pos = torch.linspace(0.25, 0.5, 2 * 29).reshape(2, 29)
    env._joint_vel = torch.linspace(-0.5, -0.25, 2 * 29).reshape(2, 29)
    env._default_joint_pos = torch.zeros((2, 29))
    env._default_joint_vel = torch.zeros((2, 29))
    env._joint_default_bias = torch.zeros((2, 29))
    env._encoder_bias = torch.full((2, 29), 0.125)
    env._raw_actions = torch.linspace(-1.0, 1.0, 2 * 29).reshape(2, 29)
    env._robot_body_pos_b = torch.zeros((2, 2, 3))
    env._robot_body_ori_b = torch.zeros((2, 2, 6))

    result = env._compute_observations(corrupt=False)
    joint_start = 58 + 3 + 6 + 3 + 3
    critic_joint_start = joint_start + 2 * 29
    torch.testing.assert_close(
        result["obs"][:, joint_start : joint_start + 29],
        result["critic"][:, critic_joint_start : critic_joint_start + 29] + 0.125,
    )


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


def test_torch_mjwarp_state_store_renegotiates_and_preserves_policy_sensor_boundary() -> None:
    class Backend:
        backend_type = "mjwarp"

        def get_state_views(self, names, device=None):
            assert names == ("qpos", "qvel")
            return {
                "qpos": torch.zeros((2, 9), device=device),
                "qvel": torch.zeros((2, 8), device=device),
            }

        def get_sensor_view(self, name, device=None):
            if name == "pelvis_local_linvel":
                return torch.ones((2, 3), device=device)
            if name == "torso_gyro":
                return torch.full((2, 3), -1.0, device=device)
            prefix = name.rsplit("_", maxsplit=1)[0]
            shape = (2, 4) if "quat" in prefix else (2, 3)
            return torch.full(shape, 2.0, device=device)

    env = module.TorchG1MotionTrackingFlashSACEnv.__new__(module.TorchG1MotionTrackingFlashSACEnv)
    env.device = torch.device("cpu")
    env._state_store = module.TensorDeviceStateStore(
        backend=Backend(),  # pyright: ignore[reportArgumentType]
        device=env.device,
        num_envs=2,
        joint_qpos_ids=np.array([7, 8], dtype=np.int64),
        joint_qvel_ids=np.array([6, 7], dtype=np.int64),
        body_names=("pelvis", "torso"),
        body_ids=np.array([0, 1], dtype=np.intp),
    )

    env._read_robot_state()

    torch.testing.assert_close(env._linvel, torch.ones((2, 3)))
    torch.testing.assert_close(env._gyro, torch.full((2, 3), -1.0))
    torch.testing.assert_close(env._robot_body_pos, torch.full((2, 2, 3), 2.0))
