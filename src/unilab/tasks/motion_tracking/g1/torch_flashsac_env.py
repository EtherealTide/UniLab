"""Torch task runtime for the FlashSAC G1 motion-tracking owner.

This is a deliberately narrow GPU implementation of the canonical
``g1_motion_tracking`` FlashSAC Manager-Based contract.  Configuration,
backend construction, scene materialization, and motion loading remain cold
CPU work in their owning modules.  The per-step action transform, motion
update, observations, reward, termination, and selected-row reset run on one
CUDA device through UniSim's public tensor lifecycle.  Unsupported task or
backend terms fail closed rather than falling back silently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from unisim.backend.base import SimBackend, TensorExecution

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import ABEnv, EnvPlayCapabilities
from unilab.base.cpu_runtime import apply_env_cpu_runtime
from unilab.base.entity import EntityCfg
from unilab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv,
    ManagerBasedRlEnvCfg,
    _resolve_backend_entity_contract,
)
from unilab.managers._noise.noise_cfg import UniformNoiseCfg
from unilab.managers.observation_manager import ObservationGroupCfg
from unilab.managers.reward_manager import RewardTermCfg
from unilab.tasks.motion_tracking.common.manager_terms import (
    MotionCommand,
    MotionCommandCfg,
    MotionJointPositionAction,
    MotionJointPositionActionCfg,
)


@dataclass
class TorchEnvState:
    """Tensor-native counterpart of UniLab's vectorized state object."""

    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]
    final_observation: dict[str, torch.Tensor] | None = None

    def replace(self, **updates: torch.Tensor | dict[str, Any]) -> TorchEnvState:
        values = vars(self).copy()
        values.update(updates)
        return TorchEnvState(**values)


def _to_device(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=value.dtype, order="C", copy=True)).to(device)


def _qualified_name(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _semantic_token(value: Any) -> Any:
    """Normalize an owner configuration value without allowing unknown escapes."""
    if value is None:
        return "none"
    if isinstance(value, slice):
        return "slice", value.start, value.stop, value.step
    if is_dataclass(value) and not isinstance(value, type):
        excluded = {"fixed_variant_plan"} if type(value).__name__ == "SceneCfg" else set()
        return (
            _qualified_name(value),
            tuple(
                (item.name, _semantic_token(getattr(value, item.name)))
                for item in dataclass_fields(value)
                if item.name not in excluded
            ),
        )
    if isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, float) and not bool(np.isfinite(value)):
            raise ValueError("Torch G1 FlashSAC owner identity contains a non-finite scalar")
        return type(value).__name__, value
    if isinstance(value, np.generic):
        if isinstance(value, (float, np.floating)) and not bool(np.isfinite(value)):
            raise ValueError("Torch G1 FlashSAC owner identity contains a non-finite scalar")
        return type(value).__name__, value.item()
    if isinstance(value, np.ndarray):
        return _qualified_name(value), str(value.dtype), value.shape, value.tobytes(order="C")
    if isinstance(value, Mapping):
        return tuple((str(key), _semantic_token(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_semantic_token(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_semantic_token(item) for item in value), key=repr))
    if callable(value):
        return "callable", _qualified_name(value)
    raise TypeError(f"unsupported Torch G1 FlashSAC owner identity value: {_qualified_name(value)}")


def _torch_g1_flashsac_owner_identity(cfg: ManagerBasedRlEnvCfg) -> str:
    """Return a versioned semantic fingerprint for the narrow Torch owner."""
    if cfg.fixed_model_variants is not None:
        raise ValueError("Torch G1 FlashSAC v1 does not support fixed model variants")
    if cfg.scene is None:
        raise ValueError("Torch G1 FlashSAC requires a scene owner")
    token = _semantic_token(
        (
            1,
            cfg.scene,
            cfg.sim_dt,
            cfg.ctrl_dt,
            cfg.max_episode_seconds,
            cfg.observations,
            cfg.actions,
            cfg.commands,
            cfg.rewards,
            cfg.terminations,
            cfg.events,
            cfg.curriculum,
            cfg.metrics,
            cfg.recorders,
            cfg.auto_reset,
            cfg.is_finite_horizon,
            cfg.scale_rewards_by_dt,
            cfg.policy_observation_group,
            cfg.critic_observation_group,
        )
    )
    return hashlib.sha256(repr(token).encode("utf-8")).hexdigest()


_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1 = (
    "6a0f2a5b731e6f7cc9f1f0eb2cb6414357f4f619bdc94bebecaff80a74161ad9"
)


def _validate_torch_g1_flashsac_owner_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    identity = _torch_g1_flashsac_owner_identity(cfg)
    if identity != _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1:
        raise ValueError(
            "Torch G1 FlashSAC v1 supports only the canonical owner contract; "
            f"semantic identity {identity} != {_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V1}"
        )


def _all_finite(*values: torch.Tensor) -> bool:
    valid = torch.isfinite(values[0]).all()
    for value in values[1:]:
        valid = valid & torch.isfinite(value).all()
    return bool(valid)


def _scalar_noise_bound(value: Any, *, label: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"FlashSAC G1 Torch runtime requires scalar {label}")
    return float(value)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., 0:1] * t + torch.linalg.cross(qv, t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_inv(q), v)


def _quat_error_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = _quat_mul(_quat_inv(a), b)
    xyz = torch.linalg.vector_norm(rel[..., 1:4], dim=-1)
    angle = 2.0 * torch.atan2(xyz, rel[..., 0].abs().clamp(max=1.0))
    return angle.square()


def _gravity_z_in_body(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2) - 1.0


def _adaptive_failure_counts(
    bin_indices: torch.Tensor, terminated: torch.Tensor, n_bins: int
) -> torch.Tensor:
    discard_bin = n_bins
    failed_bins = torch.where(
        terminated,
        bin_indices,
        torch.full_like(bin_indices, discard_bin),
    )
    return torch.bincount(failed_bins, minlength=discard_bin + 1)[:n_bins].to(torch.float32)


def _adaptive_failure_alpha(terminated: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.any(terminated).to(dtype=torch.float32) * alpha


def _rot6(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )


def _yaw_quat(robot_q: torch.Tensor, motion_q: torch.Tensor) -> torch.Tensor:
    delta = _quat_mul(robot_q, _quat_inv(motion_q))
    yaw = torch.atan2(
        2.0 * (delta[..., 0] * delta[..., 3] + delta[..., 1] * delta[..., 2]),
        1.0 - 2.0 * (delta[..., 2] * delta[..., 2] + delta[..., 3] * delta[..., 3]),
    )
    half = 0.5 * yaw
    return torch.stack(
        (torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1
    )


def _euler_xyz_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


class TorchG1MotionTrackingFlashSACEnv(ABEnv):
    """GPU runtime for the exact canonical G1 FlashSAC owner contract."""

    is_vector_env = True

    _qpos: torch.Tensor
    _qvel: torch.Tensor
    _joint_pos: torch.Tensor
    _joint_vel: torch.Tensor
    _linvel: torch.Tensor
    _gyro: torch.Tensor
    _terminations: dict[str, dict[str, Any]]
    _observation_noise_lower: torch.Tensor
    _observation_noise_upper: torch.Tensor
    _mjwarp_state_views: Mapping[str, torch.Tensor] | None
    _mjwarp_sensor_views: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] | None
    _mjwarp_linvel_view: torch.Tensor | None
    _mjwarp_gyro_view: torch.Tensor | None

    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        backend: SimBackend,
        num_envs: int,
        *,
        device: str | torch.device = "cuda",
    ):
        import torch

        self._torch = torch
        self._cfg = cfg
        self._backend = backend
        self._num_envs = int(num_envs)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device("cuda", index=torch.cuda.current_device())
        self.device = requested_device
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TorchG1MotionTrackingFlashSACEnv requires CUDA")
        if backend.num_envs != self._num_envs:
            raise ValueError("backend num_envs does not match the environment")
        self._validate_backend()
        self._mjwarp_state_views = None
        self._mjwarp_sensor_views = None
        self._mjwarp_linvel_view = None
        self._mjwarp_gyro_view = None

        self._cpu_env = ManagerBasedRlEnv(cfg, backend, self._num_envs)
        try:
            self._extract_contract()
            self._obs_groups_spec = {
                "obs": 160,
                "critic": 289,
            }
            self._observation_space = self._cpu_env.observation_space
            self._action_space = self._cpu_env.action_space
            self._state: TorchEnvState | None = None
            self._initial_seed = cfg.seed
            self._last_backend_reset_result: dict | None = None
        except BaseException:
            self._cpu_env.close()
            raise

    def _validate_backend(self) -> None:
        backend_type = self._backend.backend_type
        if backend_type not in {"mjwarp", "mujoco"}:
            raise ValueError(
                "TorchG1MotionTrackingFlashSACEnv supports mjwarp and mujoco only; "
                f"received {backend_type!r}"
            )
        mode = self._backend.tensor_execution()
        expected = (
            TensorExecution.DEVICE_RESIDENT
            if backend_type == "mjwarp"
            else TensorExecution.HOST_BRIDGE
        )
        if mode is not expected:
            raise RuntimeError(
                f"backend {backend_type!r} tensor execution is {mode!r}, expected {expected!r}"
            )
        capabilities = self._backend.get_tensor_capabilities()
        if capabilities.execution is not expected:
            raise RuntimeError(
                "backend tensor capability execution is "
                f"{capabilities.execution!r}, expected {expected!r}"
            )
        if not {"qpos", "qvel"}.issubset(capabilities.state_fields):
            missing = sorted({"qpos", "qvel"} - set(capabilities.state_fields))
            raise RuntimeError(f"backend tensor state views are missing fields: {missing}")
        if not capabilities.state_views or not capabilities.sensor_views:
            raise RuntimeError("backend did not negotiate state and sensor tensor views")
        if not capabilities.stepping or not capabilities.selected_reset:
            raise RuntimeError("backend did not negotiate tensor stepping and selected reset")
        if self._cfg.fixed_model_variants is not None:
            raise ValueError("Torch G1 FlashSAC v1 does not support fixed model variants")
        state_views = self._backend.get_state_views(("qpos", "qvel"), device=self.device)
        for name, value in state_views.items():
            if value.device != self.device:
                raise RuntimeError(
                    f"backend tensor state {name!r} lives on {value.device}, expected {self.device}"
                )
        command_cfg = self._cfg.commands.get("motion")
        if not isinstance(command_cfg, MotionCommandCfg):
            raise TypeError("Torch G1 FlashSAC requires MotionCommandCfg for sensor preflight")
        required_sensors = ["pelvis_local_linvel", "torso_gyro"]
        if backend_type == "mjwarp":
            required_sensors.extend(
                f"{prefix}_{name}"
                for name in command_cfg.body_names
                for prefix in (
                    "track_pos_w",
                    "track_quat_w",
                    "track_linvel_w",
                    "track_angvel_w",
                )
            )
        for name in required_sensors:
            value = self._backend.get_sensor_view(name, device=self.device)
            if value.device != self.device:
                raise RuntimeError(
                    f"backend tensor sensor {name!r} lives on {value.device}, "
                    f"expected {self.device}"
                )
        self._backend.get_body_ids(command_cfg.body_names)

    def _extract_contract(self) -> None:
        cfg = self._cfg
        if set(cfg.actions) != {"joint_pos"}:
            raise ValueError(f"unsupported FlashSAC G1 actions: {sorted(cfg.actions)}")
        action_cfg = cfg.actions["joint_pos"]
        if not isinstance(action_cfg, MotionJointPositionActionCfg):
            raise TypeError("FlashSAC G1 action must be MotionJointPositionActionCfg")
        if action_cfg.command_name != "motion" or action_cfg.simulate_action_latency:
            raise ValueError("unsupported FlashSAC G1 action parameters")
        if set(cfg.commands) != {"motion"} or not isinstance(
            cfg.commands["motion"], MotionCommandCfg
        ):
            raise TypeError("FlashSAC G1 requires exactly one canonical MotionCommandCfg")
        command_cfg = cfg.commands["motion"]
        if command_cfg.sampling_mode != "adaptive" or not command_cfg.params.truncate_on_clip_end:
            raise ValueError("FlashSAC G1 Torch runtime requires adaptive, truncating sampling")
        if cfg.events or cfg.curriculum or cfg.metrics or cfg.recorders:
            raise ValueError("FlashSAC G1 Torch runtime does not support extra lifecycle managers")
        if not cfg.auto_reset or cfg.is_finite_horizon:
            raise ValueError("FlashSAC G1 Torch runtime requires auto-reset infinite horizon")
        if not np.array_equal(
            self._cpu_env.scene.env_origins, np.zeros_like(self._cpu_env.scene.env_origins)
        ):
            raise ValueError("FlashSAC G1 Torch runtime requires zero scene env_origins")

        actor_group = cfg.observations.get("actor")
        critic_group = cfg.observations.get("critic")
        if not isinstance(actor_group, ObservationGroupCfg) or not isinstance(
            critic_group, ObservationGroupCfg
        ):
            raise TypeError("FlashSAC G1 requires actor and critic observation groups")
        actor_terms = tuple(actor_group.terms)
        critic_terms = tuple(critic_group.terms)
        if any(term is None for term in actor_terms) or any(term is None for term in critic_terms):
            raise ValueError("FlashSAC G1 observation terms must all be concrete")
        expected_actor = (
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        )
        expected_critic = (*expected_actor, "body_pos", "body_ori", "sac_base_lin_vel")
        if actor_terms != expected_actor or critic_terms != expected_critic:
            raise ValueError("unsupported FlashSAC G1 observation declaration")

        expected_rewards = {
            "motion_global_root_pos": 1.0,
            "motion_global_root_ori": 0.5,
            "motion_body_pos": 2.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "motion_joint_pos": 0.0,
            "motion_joint_vel": 0.0,
            "action_rate_l2": -0.1,
            "joint_limit": -2.0,
            "undesired_contacts": -0.1,
        }
        reward_terms: dict[str, RewardTermCfg] = {}
        for name, term in cfg.rewards.items():
            if not isinstance(term, RewardTermCfg):
                raise TypeError("FlashSAC G1 rewards must all be concrete RewardTermCfg terms")
            reward_terms[name] = term
        actual_rewards = {name: float(term.weight) for name, term in reward_terms.items()}
        if actual_rewards != expected_rewards:
            raise ValueError(f"unsupported FlashSAC G1 rewards: {actual_rewards}")
        if set(cfg.terminations) != {
            "time_out",
            "motion_clip_end",
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
        }:
            raise ValueError(f"unsupported FlashSAC G1 terminations: {sorted(cfg.terminations)}")
        if any(term is None for term in cfg.terminations.values()):
            raise ValueError("FlashSAC G1 terminations must all be concrete termination terms")

        cpu_command = self._cpu_env.command_manager.get_term("motion")
        cpu_action = self._cpu_env.action_manager.get_term("joint_pos")
        if not isinstance(cpu_command, MotionCommand):
            raise TypeError("cold Manager-Based construction did not build MotionCommand")
        if not isinstance(cpu_action, MotionJointPositionAction):
            raise TypeError("cold Manager-Based construction did not build motion action")
        robot = self._cpu_env.scene[command_cfg.entity_name]
        self._command = cpu_command
        self._action_cfg = action_cfg
        self._command_cfg = command_cfg
        self._robot = robot
        self._anchor_idx = command_cfg.body_names.index(command_cfg.anchor_body_name)
        self._body_names = tuple(command_cfg.body_names)
        self._body_ids = np.asarray(self._backend.get_body_ids(self._body_names), dtype=np.intp)
        self._joint_qpos_ids = np.asarray(
            self._backend.get_joint_state_qpos_indices(robot.joint_names), dtype=np.int64
        )
        self._joint_qvel_ids = np.asarray(
            self._backend.get_joint_state_qvel_indices(robot.joint_names), dtype=np.int64
        )
        target_ids = np.asarray(cpu_action._target_ids, dtype=np.int64)
        local_actuators = np.asarray(robot._joint_to_actuator_local, dtype=np.int64)[target_ids]
        self._action_to_actuator = np.asarray(robot._actuator_ids, dtype=np.int64)[local_actuators]
        self._identity_action_map = np.array_equal(
            self._action_to_actuator, np.arange(self._action_to_actuator.size, dtype=np.int64)
        )
        assert cfg.scene is not None
        robot_cfg = cfg.scene.entities["robot"]
        if not isinstance(robot_cfg, EntityCfg):
            raise TypeError("FlashSAC G1 robot scene declaration must be EntityCfg")
        if command_cfg.body_names[0] != robot_cfg.root_body_name:
            raise ValueError("canonical G1 motion body 0 must be the root body")

        self._terminations = {
            name: dict(term.params) for name, term in cfg.terminations.items() if term is not None
        }
        timeout_cfg = cfg.terminations["time_out"]
        clip_end_cfg = cfg.terminations["motion_clip_end"]
        if timeout_cfg is None or clip_end_cfg is None:
            raise ValueError("canonical FlashSAC timeout termination terms are required")
        self._terminations["time_out"]["time_out"] = bool(timeout_cfg.time_out)
        self._terminations["motion_clip_end"]["time_out"] = bool(clip_end_cfg.time_out)
        ee_names = tuple(self._terminations["ee_body_pos"]["body_names"])
        self._ee_ids = torch.tensor(
            [self._body_names.index(name) for name in ee_names],
            device=self.device,
            dtype=torch.int64,
        )
        undesired_names = (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "right_hip_roll_link",
            "right_knee_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
        )
        self._undesired_ids = torch.tensor(
            [self._body_names.index(name) for name in undesired_names],
            device=self.device,
            dtype=torch.int64,
        )
        if cfg.max_episode_seconds is None:
            raise ValueError("FlashSAC G1 requires max_episode_seconds")
        self._max_episode_length = int(round(cfg.max_episode_seconds / cfg.ctrl_dt))
        if isinstance(action_cfg.scale, dict) or isinstance(action_cfg.scale, bool):
            raise TypeError("FlashSAC G1 Torch runtime requires a scalar action scale")
        self._action_scale = float(action_cfg.scale)
        self._actor_corruption = actor_group.enable_corruption
        noise_terms: list[UniformNoiseCfg] = []
        for name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"):
            noise_term = actor_group.terms[name]
            if noise_term is None:
                raise ValueError("FlashSAC G1 noisy observation terms are required")
            noise = noise_term.noise
            if not isinstance(noise, UniformNoiseCfg):
                raise TypeError(
                    f"FlashSAC G1 noisy observations require UniformNoiseCfg: {type(noise)}"
                )
            noise_terms.append(noise)
        noise_widths = (3, 3, 29, 29)
        self._observation_noise_lower = torch.tensor(
            tuple(_scalar_noise_bound(term.n_min, label="noise minimum") for term in noise_terms),
            device=self.device,
        ).repeat_interleave(torch.tensor(noise_widths, device=self.device))
        self._observation_noise_upper = torch.tensor(
            tuple(_scalar_noise_bound(term.n_max, label="noise maximum") for term in noise_terms),
            device=self.device,
        ).repeat_interleave(torch.tensor(noise_widths, device=self.device))

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self._cfg

    @property
    def state(self) -> TorchEnvState | None:
        return self._state

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return self._obs_groups_spec

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        return EnvPlayCapabilities()

    def init_state(self) -> TorchEnvState:
        if self._state is not None:
            return self._state
        self._cpu_env.reset(seed=self._initial_seed)
        self._upload_cold_state()
        self._refresh_motion_buffers(self.current_frames)
        self._read_robot_state()
        self._refresh_relative_transforms()
        obs = self._compute_observations()
        self._state = TorchEnvState(
            obs=obs,
            reward=torch.zeros(self._num_envs, device=self.device),
            terminated=torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(self._num_envs, dtype=torch.bool, device=self.device),
            info={
                "log": {},
                "steps": torch.zeros(self._num_envs, dtype=torch.int64, device=self.device),
            },
        )
        return self._state

    def _upload_cold_state(self) -> None:
        torch = self._torch
        command = self._command
        motion = command.motion
        robot_data = self._robot.data
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(self._initial_seed or 0))

        motion_fields = tuple(
            _to_device(value, self.device).reshape(value.shape[0], -1)
            for value in (
                motion.joint_pos,
                motion.joint_vel,
                motion.body_pos_w,
                motion.body_quat_w,
                motion.body_lin_vel_w,
                motion.body_ang_vel_w,
            )
        )
        self._motion_features = torch.cat(motion_fields, dim=1)
        self.current_frames = _to_device(command.sampler.current_frames, self.device)
        self._clip_ends = _to_device(command.sampler.current_clip_end_frames, self.device)
        self._bin_failed = torch.zeros(command.sampler.bin_count, device=self.device)
        self._adaptive_kernel = _to_device(command.sampler.kernel, self.device)
        self._clip_offsets = _to_device(command.motion.clip_offsets, self.device)
        self._clip_ends_store = _to_device(command.motion.clip_end_frames, self.device)
        self._joint_default_bias = _to_device(command.joint_default_bias, self.device)
        self._default_joint_pos = _to_device(robot_data.default_joint_pos, self.device)
        self._default_joint_vel = _to_device(robot_data.default_joint_vel, self.device)
        self._soft_limits = _to_device(robot_data.soft_joint_pos_limits, self.device)
        self._encoder_bias = _to_device(robot_data.encoder_bias, self.device)
        self._raw_actions = torch.zeros_like(self._joint_default_bias)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._steps = torch.zeros(self._num_envs, dtype=torch.int64, device=self.device)
        self._ctrl = torch.zeros(
            (self._num_envs, self._backend.num_actuators), dtype=torch.float32, device=self.device
        )
        self._pose_range = _to_device(command._pose_range, self.device)
        self._velocity_range = _to_device(command._velocity_range, self.device)
        cold_finite_tensors = (
            self._motion_features,
            self.current_frames,
            self._clip_ends,
            self._adaptive_kernel,
            self._clip_offsets,
            self._clip_ends_store,
            self._joint_default_bias,
            self._default_joint_pos,
            self._default_joint_vel,
            self._soft_limits,
            self._encoder_bias,
            self._pose_range,
            self._velocity_range,
        )
        if not _all_finite(*cold_finite_tensors):
            raise ValueError("Torch G1 FlashSAC cold state contains NaN or Inf")

        shape = (self._num_envs, len(self._body_names))
        motion_width = self._motion_features.shape[1]
        self._motion_state = torch.empty(
            (self._num_envs, motion_width), dtype=torch.float32, device=self.device
        )
        joint_width = motion_fields[0].shape[1]
        body_width = shape[1] * 3
        quat_width = shape[1] * 4
        cursor = 0
        self._motion_joint_pos = self._motion_state[:, cursor : cursor + joint_width]
        cursor += joint_width
        self._motion_joint_vel = self._motion_state[:, cursor : cursor + joint_width]
        cursor += joint_width
        self._motion_body_pos = self._motion_state[:, cursor : cursor + body_width].view(
            self._num_envs, shape[1], 3
        )
        cursor += body_width
        self._motion_body_quat = self._motion_state[:, cursor : cursor + quat_width].view(
            self._num_envs, shape[1], 4
        )
        cursor += quat_width
        self._motion_body_lin_vel = self._motion_state[:, cursor : cursor + body_width].view(
            self._num_envs, shape[1], 3
        )
        cursor += body_width
        self._motion_body_ang_vel = self._motion_state[:, cursor:motion_width].view(
            self._num_envs, shape[1], 3
        )
        self._robot_body_pos = torch.empty_like(self._motion_body_pos)
        self._robot_body_quat = torch.empty_like(self._motion_body_quat)
        self._robot_body_lin_vel = torch.empty_like(self._motion_body_lin_vel)
        self._robot_body_ang_vel = torch.empty_like(self._motion_body_ang_vel)
        self._body_pos_relative = torch.empty_like(self._motion_body_pos)
        self._body_quat_relative = torch.empty_like(self._motion_body_quat)
        self._motion_anchor_pos_b = torch.empty((self._num_envs, 3), device=self.device)
        self._motion_anchor_ori_b = torch.empty((self._num_envs, 6), device=self.device)
        self._robot_body_pos_b = torch.empty_like(self._robot_body_pos)
        self._robot_body_ori_b = torch.empty((*shape, 6), device=self.device)
        self._obs = {
            name: torch.empty((self._num_envs, dim), device=self.device)
            for name, dim in self.obs_groups_spec.items()
        }
        self._final_observation = {
            name: torch.empty_like(value) for name, value in self._obs.items()
        }

    def _motion_at(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        selected = torch.index_select(self._motion_features, 0, frames.long())
        joint_width = self._motion_joint_pos.shape[1]
        body_count = self._motion_body_pos.shape[1]
        body_width = body_count * 3
        quat_width = body_count * 4
        cursor = 0
        values = []
        for width, body_shape in (
            (joint_width, (joint_width,)),
            (joint_width, (joint_width,)),
            (body_width, (body_count, 3)),
            (quat_width, (body_count, 4)),
            (body_width, (body_count, 3)),
            (body_width, (body_count, 3)),
        ):
            values.append(selected[:, cursor : cursor + width].view(-1, *body_shape))
            cursor += width
        return tuple(values)

    def _refresh_motion_buffers(
        self, frames: torch.Tensor, rows: torch.Tensor | None = None
    ) -> None:
        selected = torch.index_select(self._motion_features, 0, frames.long())
        target = slice(None) if rows is None else rows
        self._motion_state[target] = selected

    def _read_robot_state(self, rows: torch.Tensor | None = None) -> None:
        torch = self._torch
        if self._backend.backend_type == "mjwarp":
            if self._mjwarp_state_views is None or self._mjwarp_sensor_views is None:
                self._mjwarp_state_views = self._backend.get_state_views(
                    ("qpos", "qvel"), device=self.device
                )
                linvel_view = self._backend.get_sensor_view(
                    "pelvis_local_linvel", device=self.device
                )
                gyro_view = self._backend.get_sensor_view("torso_gyro", device=self.device)
                if not isinstance(linvel_view, torch.Tensor) or not isinstance(
                    gyro_view, torch.Tensor
                ):
                    raise TypeError("MJWarp scalar sensor views did not return tensors")
                self._mjwarp_sensor_views = {
                    "linvel": linvel_view,
                    "gyro": gyro_view,
                }
                for prefix in (
                    "track_pos_w",
                    "track_quat_w",
                    "track_linvel_w",
                    "track_angvel_w",
                ):
                    self._mjwarp_sensor_views[prefix] = tuple(
                        self._backend.get_sensor_view(f"{prefix}_{name}", device=self.device)
                        for name in self._body_names
                    )
                self._mjwarp_linvel_view = linvel_view
                self._mjwarp_gyro_view = gyro_view
                self._linvel = linvel_view.clone()
                self._gyro = gyro_view.clone()
            else:
                assert self._mjwarp_linvel_view is not None
                assert self._mjwarp_gyro_view is not None
                # MJWarp's tracked-body refresh also evaluates authored frame
                # sensors at the final qpos/qvel. Preserve the legacy host-path
                # substep-boundary values for policy sensors while still using
                # refreshed tracked-body state for rewards and terminations.
                self._linvel.copy_(self._mjwarp_linvel_view)
                self._gyro.copy_(self._mjwarp_gyro_view)
                # Stable views do not negotiate lifecycle state on dereference.
                # One tracked-sensor read refreshes all injected frame sensors
                # after step/reset while avoiding per-body backend calls.
                self._backend.get_sensor_view(
                    f"track_pos_w_{self._body_names[0]}", device=self.device
                )
                if rows is not None:
                    self._linvel[rows] = self._mjwarp_linvel_view.index_select(0, rows)
                    self._gyro[rows] = self._mjwarp_gyro_view.index_select(0, rows)
            assert self._mjwarp_state_views is not None
            assert self._mjwarp_sensor_views is not None
            self._qpos = self._mjwarp_state_views["qpos"]
            self._qvel = self._mjwarp_state_views["qvel"]
            self._joint_pos = self._qpos[:, self._joint_qpos_ids]
            self._joint_vel = self._qvel[:, self._joint_qvel_ids]
            targets = (
                self._robot_body_pos,
                self._robot_body_quat,
                self._robot_body_lin_vel,
                self._robot_body_ang_vel,
            )
            prefixes = ("track_pos_w", "track_quat_w", "track_linvel_w", "track_angvel_w")
            for destination, prefix in zip(targets, prefixes, strict=True):
                views = self._mjwarp_sensor_views[prefix]
                if not isinstance(views, tuple):
                    raise TypeError("MJWarp body sensor views did not return a tuple")
                if rows is None:
                    torch.stack(views, dim=1, out=destination)
                else:
                    selected = torch.stack(
                        tuple(view.index_select(0, rows) for view in views),
                        dim=1,
                    )
                    destination.index_copy_(0, rows, selected)
            return

        if rows is None:
            state = self._backend.get_state_views(("qpos", "qvel"), device=self.device)
            self._qpos = state["qpos"]
            self._qvel = state["qvel"]
            self._joint_pos = self._qpos[:, self._joint_qpos_ids]
            self._joint_vel = self._qvel[:, self._joint_qvel_ids]
        self._joint_pos = self._qpos[:, self._joint_qpos_ids]
        self._joint_vel = self._qvel[:, self._joint_qvel_ids]
        if rows is None:
            self._linvel = self._backend.get_sensor_view("pelvis_local_linvel", device=self.device)
            self._gyro = self._backend.get_sensor_view("torso_gyro", device=self.device)
        else:
            host_rows = rows.detach().cpu().numpy()
            linvel = self._backend.get_sensor_data_rows("pelvis_local_linvel", host_rows)
            gyro = self._backend.get_sensor_data_rows("torso_gyro", host_rows)
            self._linvel[rows] = torch.as_tensor(
                np.ascontiguousarray(linvel), dtype=torch.float32, device=self.device
            )
            self._gyro[rows] = torch.as_tensor(
                np.ascontiguousarray(gyro), dtype=torch.float32, device=self.device
            )
        if rows is None:
            self._robot_body_pos.copy_(
                torch.as_tensor(
                    np.ascontiguousarray(self._backend.get_body_pos_w(self._body_ids)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self._robot_body_quat.copy_(
                torch.as_tensor(
                    np.ascontiguousarray(self._backend.get_body_quat_w(self._body_ids)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self._robot_body_lin_vel.copy_(
                torch.as_tensor(
                    np.ascontiguousarray(self._backend.get_body_lin_vel_w(self._body_ids)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self._robot_body_ang_vel.copy_(
                torch.as_tensor(
                    np.ascontiguousarray(self._backend.get_body_ang_vel_w(self._body_ids)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        else:
            host_rows = rows.detach().cpu().numpy()
            pos, quat = self._backend.get_body_pose_w_rows(host_rows, self._body_ids)
            lin_vel = self._backend.get_body_lin_vel_w_rows(host_rows, self._body_ids)
            ang_vel = self._backend.get_body_ang_vel_w_rows(host_rows, self._body_ids)
            self._robot_body_pos[rows] = torch.as_tensor(
                np.ascontiguousarray(pos), dtype=torch.float32, device=self.device
            )
            self._robot_body_quat[rows] = torch.as_tensor(
                np.ascontiguousarray(quat), dtype=torch.float32, device=self.device
            )
            self._robot_body_lin_vel[rows] = torch.as_tensor(
                np.ascontiguousarray(lin_vel), dtype=torch.float32, device=self.device
            )
            self._robot_body_ang_vel[rows] = torch.as_tensor(
                np.ascontiguousarray(ang_vel), dtype=torch.float32, device=self.device
            )

    def _refresh_motion_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        target = slice(None) if rows is None else rows
        anchor = self._anchor_idx
        m_anchor_pos = self._motion_body_pos[target, anchor]
        m_anchor_quat = self._motion_body_quat[target, anchor]
        r_anchor_pos = self._robot_body_pos[target, anchor]
        r_anchor_quat = self._robot_body_quat[target, anchor]
        delta = _yaw_quat(r_anchor_quat, m_anchor_quat)
        rotated_body_quat = _quat_mul(
            delta[None, :] if delta.ndim == 1 else delta[:, None, :], self._motion_body_quat[target]
        )
        self._body_quat_relative[target] = rotated_body_quat
        local = self._motion_body_pos[target] - m_anchor_pos[:, None, :]
        rotated_local = _quat_apply(delta[None, :] if delta.ndim == 1 else delta[:, None, :], local)
        self._body_pos_relative[target] = rotated_local + r_anchor_pos[:, None, :]
        relative = self._body_pos_relative[target]
        relative[..., 2] = self._motion_body_pos[target][..., 2]
        self._body_pos_relative[target] = relative
        anchor_delta = m_anchor_pos - r_anchor_pos
        self._motion_anchor_pos_b[target] = _quat_apply_inverse(r_anchor_quat, anchor_delta)
        self._motion_anchor_ori_b[target] = _rot6(
            _quat_mul(_quat_inv(r_anchor_quat), m_anchor_quat)
        )

    def _refresh_robot_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        target = slice(None) if rows is None else rows
        anchor = self._anchor_idx
        r_anchor_pos = self._robot_body_pos[target, anchor]
        r_anchor_quat = self._robot_body_quat[target, anchor]
        robot_local = self._robot_body_pos[target] - r_anchor_pos[:, None, :]
        self._robot_body_pos_b[target] = _quat_apply_inverse(r_anchor_quat[:, None, :], robot_local)
        self._robot_body_ori_b[target] = _rot6(
            _quat_mul(_quat_inv(r_anchor_quat[:, None, :]), self._robot_body_quat[target])
        )

    def _refresh_relative_transforms(self, rows: torch.Tensor | None = None) -> None:
        self._refresh_motion_relative_transforms(rows)
        self._refresh_robot_relative_transforms(rows)

    def _exp_error(self, error: torch.Tensor, std: float) -> torch.Tensor:
        return torch.exp(error / (-(std * std)))

    def _compute_terminations(self) -> torch.Tensor:
        anchor_cfg = self._terminations["anchor_pos"]
        ee_cfg = self._terminations["ee_body_pos"]
        anchor_idx = self._anchor_idx
        anchor_pos_bad = (
            self._motion_body_pos[:, anchor_idx, 2] - self._robot_body_pos[:, anchor_idx, 2]
        ).abs() > float(anchor_cfg["threshold"])
        anchor_ori_cfg = self._terminations["anchor_ori"]
        motion_z = _gravity_z_in_body(self._motion_body_quat[:, anchor_idx])
        robot_z = _gravity_z_in_body(self._robot_body_quat[:, anchor_idx])
        anchor_ori_bad = (motion_z - robot_z).abs() > float(anchor_ori_cfg["threshold"])
        ee_bad = torch.any(
            (
                self._body_pos_relative[:, self._ee_ids, 2]
                - self._robot_body_pos[:, self._ee_ids, 2]
            ).abs()
            > float(ee_cfg["threshold"]),
            dim=-1,
        )
        return anchor_pos_bad | anchor_ori_bad | ee_bad

    def _compute_reward(self) -> torch.Tensor:
        anchor = self._anchor_idx
        terms: list[torch.Tensor] = []
        self.last_reward_terms: dict[str, torch.Tensor] = {}
        pos_error = (
            (self._motion_body_pos[:, anchor] - self._robot_body_pos[:, anchor])
            .square()
            .sum(dim=-1)
        )
        root_pos_term = self._exp_error(pos_error, 0.3)
        self.last_reward_terms["motion_global_root_pos"] = root_pos_term
        terms.append(root_pos_term)
        ori_error = _quat_error_squared(
            self._motion_body_quat[:, anchor], self._robot_body_quat[:, anchor]
        )
        root_ori_term = self._exp_error(ori_error, 0.4)
        self.last_reward_terms["motion_global_root_ori"] = root_ori_term
        terms.append(0.5 * root_ori_term)
        body_specs = (
            ("motion_body_pos", 2.0, 0.3, self._body_pos_relative, self._robot_body_pos),
            ("motion_body_ori", 1.0, 0.4, self._body_quat_relative, self._robot_body_quat),
            ("motion_body_lin_vel", 1.0, 1.0, self._motion_body_lin_vel, self._robot_body_lin_vel),
            ("motion_body_ang_vel", 1.0, 3.14, self._motion_body_ang_vel, self._robot_body_ang_vel),
        )
        for name, weight, std, reference, actual in body_specs:
            body_std = std * (reference.shape[1] ** 0.5)
            if reference is self._body_quat_relative:
                error = _quat_error_squared(reference, actual).sum(dim=-1)
            else:
                error = (reference - actual).square().sum(dim=(-1, -2))
            term = self._exp_error(error, float(body_std))
            self.last_reward_terms[name] = term
            terms.append(weight * term)
        action_rate = (self._raw_actions - self._previous_raw_actions).square().sum(dim=-1)
        self.last_reward_terms["action_rate_l2"] = action_rate
        terms.append(-0.1 * action_rate)
        lower_violation = (self._soft_limits[..., 0] - self._joint_pos).clamp_min(0.0)
        upper_violation = (self._joint_pos - self._soft_limits[..., 1]).clamp_min(0.0)
        joint_violation = (lower_violation + upper_violation).square().sum(dim=-1)
        self.last_reward_terms["joint_limit"] = joint_violation
        terms.append(-2.0 * joint_violation)
        undesired_cfg = self._terminations  # body names are in the reward owner, not terminations
        del undesired_cfg
        contacts = (self._robot_body_pos[:, self._undesired_ids, 2] < 0.05).sum(dim=-1)
        contact_term = contacts.to(torch.float32)
        self.last_reward_terms["undesired_contacts"] = contact_term
        terms.append(-0.1 * contact_term)
        reward = terms[0]
        for term in terms[1:]:
            reward = reward + term
        return reward * self._cfg.ctrl_dt

    def _compute_observations(
        self, *, corrupt: bool | None = None, rows: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        target = slice(None) if rows is None else rows
        command = torch.cat(
            (self._motion_joint_pos[target], self._motion_joint_vel[target]), dim=-1
        )
        joint_pos = (
            self._joint_pos[target]
            - self._default_joint_pos[target]
            - self._joint_default_bias[target]
        )
        joint_vel = self._joint_vel[target] - self._default_joint_vel[target]
        linvel = self._linvel[target]
        gyro = self._gyro[target]
        clean_actor = torch.cat(
            (
                command,
                self._motion_anchor_pos_b[target],
                self._motion_anchor_ori_b[target],
                linvel,
                gyro,
                joint_pos,
                joint_vel,
                self._raw_actions[target],
            ),
            dim=-1,
        )
        actor = clean_actor.clone()
        noisy = self._actor_corruption if corrupt is None else corrupt
        if noisy:
            cursor = 58 + 3 + 6
            width = self._observation_noise_lower.numel()
            noise = torch.rand((*actor.shape[:-1], width), device=self.device, generator=self._rng)
            noise *= self._observation_noise_upper - self._observation_noise_lower
            noise += self._observation_noise_lower
            actor[..., cursor : cursor + width] += noise
        critic = torch.cat(
            (
                clean_actor,
                self._robot_body_pos_b[target].reshape(target_length := actor.shape[0], -1),
                self._robot_body_ori_b[target].reshape(target_length, -1),
                linvel,
            ),
            dim=-1,
        )
        return {"obs": actor, "critic": critic}

    def _sample_frames(self, rows: torch.Tensor) -> torch.Tensor:
        probs = (
            self._bin_failed
            + float(self._command_cfg.params.adaptive_uniform_ratio) / self._bin_failed.numel()
        )
        kernel_size = int(self._command_cfg.params.adaptive_kernel_size)
        if kernel_size > 1:
            probs = torch.nn.functional.pad(probs, (0, kernel_size - 1), mode="replicate")
            probs = torch.nn.functional.conv1d(
                probs[None, None, :], self._adaptive_kernel[None, None, :], padding=0
            )[0, 0]
        probs = probs / probs.sum()
        bins = torch.multinomial(probs, rows.numel(), replacement=True, generator=self._rng)
        offsets = torch.rand(rows.numel(), device=self.device, generator=self._rng)
        frames = (
            (bins.to(torch.float32) + offsets)
            / self._bin_failed.numel()
            * (self._motion_features.shape[0] - 1)
        ).to(torch.int32)
        self.current_frames[rows] = frames
        clip_indices = (
            torch.searchsorted(
                self._clip_offsets.int(),
                frames.int(),
                right=True,
            )
            - 1
        )
        clip_indices = clip_indices.clamp_(min=0)
        self._clip_ends[rows] = self._clip_ends_store[clip_indices]
        return frames

    def _reset_rows(
        self, rows: torch.Tensor, state_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        torch = self._torch
        if rows.numel() == 0:
            return state_obs
        frames = self._sample_frames(rows)
        motion = self._motion_at(frames)
        count = rows.numel()
        pose = torch.rand((count, 6), device=self.device, generator=self._rng)
        pose = (
            pose * (self._pose_range[None, :, 1] - self._pose_range[None, :, 0])
            + self._pose_range[None, :, 0]
        )
        velocity = torch.rand((count, 6), device=self.device, generator=self._rng)
        velocity = (
            velocity * (self._velocity_range[None, :, 1] - self._velocity_range[None, :, 0])
            + self._velocity_range[None, :, 0]
        )
        root_pos = motion[2][:, 0] + pose[:, :3]
        root_quat = _quat_mul(_euler_xyz_quat(pose[:, 3], pose[:, 4], pose[:, 5]), motion[3][:, 0])
        root_lin_vel = motion[4][:, 0] + velocity[:, :3]
        root_ang_vel = motion[5][:, 0] + velocity[:, 3:]
        joint_range = self._command_cfg.params.joint_position_range
        joint_noise_scale = float(joint_range[1] - joint_range[0])
        joint_noise = torch.rand(
            (count, self._motion_joint_pos.shape[1]),
            device=self.device,
            generator=self._rng,
        ) * joint_noise_scale + float(joint_range[0])
        joint_pos = motion[0] + joint_noise
        joint_pos = joint_pos.clamp(self._soft_limits[:, 0], self._soft_limits[:, 1])
        qpos = self._qpos.index_select(0, rows).clone()
        qvel = self._qvel.index_select(0, rows).clone()
        qpos[:, :3] = root_pos
        qpos[:, 3:7] = root_quat
        qpos[:, self._joint_qpos_ids] = joint_pos
        qvel[:, :3] = root_lin_vel
        qvel[:, 3:6] = root_ang_vel
        qvel[:, self._joint_qvel_ids] = motion[1]
        if not _all_finite(qpos, qvel):
            raise ValueError("Torch G1 FlashSAC reset qpos/qvel contain NaN or Inf")
        self._last_backend_reset_result = self._backend.set_state_tensor(rows, qpos, qvel)
        self._qpos[rows] = qpos
        self._qvel[rows] = qvel
        default_range = self._command_cfg.params.joint_default_position_range
        default_noise_scale = float(default_range[1] - default_range[0])
        self._joint_default_bias[rows] = torch.rand(
            (count, self._joint_default_bias.shape[1]), device=self.device, generator=self._rng
        ) * default_noise_scale + float(default_range[0])
        self._raw_actions[rows] = 0.0
        self._previous_raw_actions[rows] = 0.0
        self._ctrl[rows] = 0.0
        self._steps[rows] = 0
        self._refresh_motion_buffers(frames, rows)
        self._read_robot_state(rows)
        self._refresh_motion_relative_transforms(rows)
        self._refresh_robot_relative_transforms(rows)
        reset_obs = self._compute_observations(rows=rows)
        for name in state_obs:
            state_obs[name][rows] = reset_obs[name]
        return state_obs

    def step(self, actions: torch.Tensor | np.ndarray) -> TorchEnvState:
        torch = self._torch
        if self._state is None:
            raise RuntimeError("call init_state() before step()")
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action.shape != self._raw_actions.shape:
            raise ValueError(
                f"expected action shape {tuple(self._raw_actions.shape)}, got {tuple(action.shape)}"
            )

        timing: dict[str, float] = {}
        started = perf_counter()
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(action)
        processed = self._raw_actions * self._action_scale
        if self._action_cfg.use_default_offset:
            processed = processed + self._default_joint_pos
        target = processed + self._joint_default_bias - self._encoder_bias
        if self._identity_action_map:
            self._ctrl.copy_(target)
        else:
            self._ctrl.zero_()
            self._ctrl[:, self._action_to_actuator] = target
        if not _all_finite(action, self._ctrl):
            if not bool(torch.isfinite(action).all()):
                raise ValueError("actions contain NaN or Inf")
            raise ValueError("transformed controls contain NaN or Inf")
        backend_result = self._backend.step_tensor(self._ctrl, nsteps=self._cfg.sim_substeps)
        if isinstance(backend_result, dict):
            timing.update(backend_result.get("timing", {}))
        timing["action_backend_step_ms"] = (perf_counter() - started) * 1000.0

        phase = perf_counter()
        self._steps += 1
        self._read_robot_state()
        self._refresh_robot_relative_transforms()
        terminated = self._compute_terminations()
        clip_end = self.current_frames >= self._clip_ends
        timeout = self._steps >= self._max_episode_length
        truncated = clip_end | timeout
        reward = self._compute_reward()

        bin_indices = (
            self.current_frames.to(torch.int64)
            * self._bin_failed.numel()
            // self._motion_features.shape[0]
        ).clamp_(max=self._bin_failed.numel() - 1)
        n_bins = self._bin_failed.numel()
        failures = _adaptive_failure_counts(bin_indices, terminated, n_bins)
        failure_alpha = _adaptive_failure_alpha(
            terminated, self._command_cfg.params.adaptive_alpha
        ).to(dtype=self._bin_failed.dtype)
        self._bin_failed.mul_(1.0 - failure_alpha).add_(failures * failure_alpha)
        active = ~(terminated | truncated)
        self.current_frames += active.to(torch.int32)
        self._refresh_motion_buffers(self.current_frames)
        self._refresh_motion_relative_transforms()
        obs = self._compute_observations()
        timing["update_state_ms"] = (perf_counter() - phase) * 1000.0

        phase = perf_counter()
        done = terminated | truncated
        final_obs = None
        rows = done.nonzero(as_tuple=False).flatten().to(torch.int64)
        if rows.numel():
            final_obs = self._final_observation
            for name, values in obs.items():
                final_obs[name].index_copy_(0, rows, values.index_select(0, rows))
        obs = self._reset_rows(rows, obs)
        if rows.numel() and isinstance(self._last_backend_reset_result, dict):
            timing.update(self._last_backend_reset_result.get("timing", {}))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timing["reset_done_ms"] = (perf_counter() - phase) * 1000.0
        timing["step_ms"] = (perf_counter() - started) * 1000.0

        self._state = TorchEnvState(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"log": {}, "steps": self._steps.clone(), "timing": timing},
            final_observation=final_obs,
        )
        return self._state

    def reset(
        self,
        env_indices: torch.Tensor | np.ndarray | None = None,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | np.ndarray | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del options
        if env_indices is not None and env_ids is not None:
            raise ValueError("pass either env_indices or env_ids, not both")
        source = env_indices if env_indices is not None else env_ids
        if seed is not None:
            self._initial_seed = seed
            if self._state is not None:
                self._rng.manual_seed(seed)
        if self._state is None:
            state = self.init_state()
            return {name: value.clone() for name, value in state.obs.items()}, {"log": {}}
        rows = (
            torch.arange(self._num_envs, device=self.device, dtype=torch.int64)
            if source is None
            else torch.as_tensor(source, dtype=torch.int64, device=self.device)
        )
        obs = self._reset_rows(rows, {k: v.clone() for k, v in self._state.obs.items()})
        self._state.terminated[rows] = False
        self._state.truncated[rows] = False
        return {name: values[rows].clone() for name, values in obs.items()}, {"log": {}}

    def set_nan_guard(self, guard: Any) -> None:
        self._cpu_env.set_nan_guard(guard)

    def close(self) -> None:
        self._cpu_env.close()

    cleanup = close


def make_torch_g1_motion_tracking_flashsac_env(
    cfg: ManagerBasedRlEnvCfg,
    num_envs: int = 1,
    backend_type: str = "mujoco",
) -> ABEnv:
    """Registry factory using the normal Manager-Based cold path and assets."""
    from unilab.envs import make_manager_based_rl_env

    if not isinstance(cfg, ManagerBasedRlEnvCfg):
        raise TypeError("Torch G1 FlashSAC factory expected ManagerBasedRlEnvCfg")
    if backend_type not in {"mujoco", "mjwarp"}:
        raise ValueError("Torch G1 FlashSAC runtime supports mujoco and mjwarp only")
    if not cfg.tensor_runtime:
        env = make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type=backend_type)
        return env
    _validate_torch_g1_flashsac_owner_contract(cfg)
    cfg.validate()
    assert cfg.scene is not None
    apply_env_cpu_runtime(cfg.cpu_ids)
    base_name, body_state_requested, tracked_body_names = _resolve_backend_entity_contract(cfg)
    kwargs = env_backend_kwargs(cfg)
    kwargs["base_name"] = base_name
    if backend_type == "mujoco" and tracked_body_names is not None:
        kwargs["tracked_body_names"] = tracked_body_names
    backend = create_backend(
        backend_type,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=body_state_requested,
        **kwargs,
    )
    try:
        return TorchG1MotionTrackingFlashSACEnv(cfg, backend, num_envs, device="cuda")
    except BaseException:
        backend.cleanup_scene_assets()
        raise


__all__ = [
    "TorchEnvState",
    "TorchG1MotionTrackingFlashSACEnv",
    "make_torch_g1_motion_tracking_flashsac_env",
]
