"""Minimal tensor parity runtime for the canonical Go2 joystick-flat owner.

This module is an M9 contract pilot only.  It is intentionally not registered as a
production backend and does not change any backend support claim.  The runtime
negotiates UniSim's public tensor lifecycle and keeps the canonical Go2
observation, reward, termination, and selected-reset equations tensor-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from unisim.backend.base import SimBackend, TensorExecution, tensor_device_matches

GO2_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
GO2_ACTUATOR_NAMES: tuple[str, ...] = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
_JOINT_TO_ACTUATOR: tuple[int, ...] = tuple(
    GO2_ACTUATOR_NAMES.index(name.removesuffix("_joint")) for name in GO2_JOINT_NAMES
)
_DEFAULT_JOINT_POS: tuple[float, ...] = (
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    1.0,
    -1.5,
    0.0,
    1.0,
    -1.5,
)
_FOOT_CONTACT_SENSOR_NAMES: tuple[str, ...] = (
    "FL_foot_contact",
    "FR_foot_contact",
    "RL_foot_contact",
    "RR_foot_contact",
)
_FOOT_POSITION_SENSOR_NAMES: tuple[str, ...] = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
_SENSOR_NAMES: tuple[str, ...] = (
    "gyro",
    "local_linvel",
    "upvector",
    *_FOOT_CONTACT_SENSOR_NAMES,
    *_FOOT_POSITION_SENSOR_NAMES,
)
_GAIT_OFFSETS: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0)


@dataclass(frozen=True)
class Go2TensorParityConfig:
    """Numerical parameters of the canonical DR-free Go2 parity owner."""

    action_scale: float = 0.25
    ctrl_dt: float = 0.02
    max_episode_steps: int = 1_000
    gait_frequency: float = 2.0
    tracking_lin_vel_std: float = 0.5
    tracking_ang_vel_std: float = 0.5
    base_height_target: float = 0.3
    foot_target_height: float = 0.1
    foot_height_kernel: float = 0.01
    swing_start: float = 0.6
    contact_threshold: float = 0.1
    stance_threshold: float = 0.6
    bad_orientation_limit: float = 1.0471975511965976
    tracking_lin_vel_weight: float = 1.0
    tracking_ang_vel_weight: float = 0.2
    lin_vel_z_weight: float = -5.0
    ang_vel_xy_weight: float = -0.1
    base_height_weight: float = -100.0
    action_rate_weight: float = -0.005
    joint_deviation_weight: float = -0.1
    contact_weight: float = 0.24
    swing_feet_weight: float = 4.0


@dataclass
class Go2TensorParityState:
    """Torch state returned by the Go2 parity pilot."""

    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]


class Go2TensorParityRuntime:
    """Exercise the canonical Go2 equations through public backend tensor APIs."""

    def __init__(
        self,
        backend: SimBackend,
        *,
        command: torch.Tensor,
        reset_qpos: torch.Tensor,
        reset_qvel: torch.Tensor,
        device: str | torch.device = "cpu",
        cfg: Go2TensorParityConfig = Go2TensorParityConfig(),
    ):
        self.backend = backend
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA Go2 tensor parity requested but CUDA is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", index=torch.cuda.current_device())
        self._negotiate_backend()
        self.num_envs = int(backend.num_envs)
        self.num_joints = len(GO2_JOINT_NAMES)
        if backend.num_actuators != self.num_joints:
            raise ValueError(
                f"Go2 parity backend must expose {self.num_joints} actuators; "
                f"received {backend.num_actuators}"
            )

        state_views = backend.get_state_views(("qpos", "qvel"), device=self.device)
        self._qpos = self._require_tensor(state_views.get("qpos"), "qpos")
        self._qvel = self._require_tensor(state_views.get("qvel"), "qvel")
        expected_qpos = (self.num_envs, 7 + self.num_joints)
        expected_qvel = (self.num_envs, 6 + self.num_joints)
        if self._qpos.shape != expected_qpos:
            raise ValueError(
                f"Go2 qpos view shape is {tuple(self._qpos.shape)}, not {expected_qpos}"
            )
        if self._qvel.shape != expected_qvel:
            raise ValueError(
                f"Go2 qvel view shape is {tuple(self._qvel.shape)}, not {expected_qvel}"
            )

        self.sensors = {
            name: self._require_tensor(backend.get_sensor_view(name, self.device), name)
            for name in _SENSOR_NAMES
        }
        expected_sensor_shapes = {
            "gyro": (self.num_envs, 3),
            "local_linvel": (self.num_envs, 3),
            "upvector": (self.num_envs, 3),
            **{name: (self.num_envs, 1) for name in _FOOT_CONTACT_SENSOR_NAMES},
            **{name: (self.num_envs, 3) for name in _FOOT_POSITION_SENSOR_NAMES},
        }
        for name, view in self.sensors.items():
            if view.shape != expected_sensor_shapes[name]:
                raise ValueError(
                    f"Go2 sensor {name!r} shape is {tuple(view.shape)}, "
                    f"expected {expected_sensor_shapes[name]}"
                )

        self.command = self._require_input(command, "command", (self.num_envs, 3))
        self.reset_qpos = self._require_input(
            reset_qpos, "reset_qpos", (self.num_envs, 7 + self.num_joints)
        )
        self.reset_qvel = self._require_input(
            reset_qvel, "reset_qvel", (self.num_envs, 6 + self.num_joints)
        )

        self._default_joint_pos = torch.tensor(
            _DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._default_joint_vel = torch.zeros_like(self._default_joint_pos)
        joint_to_actuator = torch.tensor(_JOINT_TO_ACTUATOR, dtype=torch.int64, device=self.device)
        self._default_ctrl = self._default_joint_pos.index_select(1, joint_to_actuator)
        self._gait_offsets = torch.tensor(
            _GAIT_OFFSETS, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._raw_actions = torch.zeros_like(self._default_ctrl)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._ctrl = torch.zeros_like(self._default_ctrl)
        self._steps = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self._phase = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._obs = {
            "policy": torch.empty((self.num_envs, 49), dtype=torch.float32, device=self.device),
            "critic": torch.empty((self.num_envs, 52), dtype=torch.float32, device=self.device),
        }
        self.reward_components = {
            name: torch.empty((self.num_envs,), dtype=torch.float32, device=self.device)
            for name in (
                "tracking_lin_vel",
                "tracking_ang_vel",
                "lin_vel_z",
                "ang_vel_xy",
                "base_height",
                "action_rate",
                "similar_to_default",
                "contact",
                "swing_feet_z",
            )
        }

    def _negotiate_backend(self) -> None:
        capabilities = self.backend.get_tensor_capabilities()
        execution = capabilities.execution
        if execution is not TensorExecution.DEVICE_RESIDENT:
            raise RuntimeError(
                "Go2 tensor parity runtime requires a DEVICE_RESIDENT backend; "
                f"received {execution!r}"
            )
        if self.backend.tensor_execution() is not execution:
            raise RuntimeError("backend tensor execution does not match its capabilities")
        missing_fields = {"qpos", "qvel"} - set(capabilities.state_fields)
        if missing_fields:
            raise RuntimeError(f"backend tensor state fields are missing: {sorted(missing_fields)}")
        if not capabilities.state_views:
            raise RuntimeError("backend did not negotiate tensor state views")
        if not capabilities.sensor_views:
            raise RuntimeError("backend did not negotiate tensor sensor views")
        if not capabilities.stepping:
            raise RuntimeError("backend did not negotiate tensor stepping")
        if not capabilities.selected_reset:
            raise RuntimeError("backend did not negotiate selected-row tensor reset")
        if not tensor_device_matches(
            capabilities.torch_devices,
            self.device,
            current_device=self.device.index,
        ):
            raise RuntimeError(
                f"backend did not accept Torch device {str(self.device)!r}; "
                f"supported devices are {capabilities.torch_devices}"
            )

    def _require_tensor(self, value: Any, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"backend tensor view {name!r} did not return a Torch tensor")
        if value.device != self.device:
            raise RuntimeError(f"backend tensor view {name!r} lives on {value.device}")
        if value.dtype is not torch.float32:
            raise TypeError(f"backend tensor view {name!r} must be float32, got {value.dtype}")
        return value

    def _require_input(self, value: Any, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
        if value.shape != shape:
            raise ValueError(f"{name} shape is {tuple(value.shape)}, expected {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} lives on {value.device}, expected {self.device}")
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains NaN or Inf")
        return value

    def _validate_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self._require_input(actions, "actions", (self.num_envs, self.num_joints))

    def _validate_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.dtype is not torch.int64 or rows.ndim != 1:
            raise TypeError("reset rows must be a one-dimensional int64 tensor")
        if rows.device != self.device:
            raise ValueError(f"reset rows live on {rows.device}, expected {self.device}")
        if not rows.is_contiguous():
            raise ValueError("reset rows must be contiguous")
        if rows.numel() == 0:
            return rows
        unique_count = torch.unique(rows).numel()
        minimum_row, maximum_row, validated_count = (
            int(value)
            for value in torch.stack(
                (
                    rows.min(),
                    rows.max(),
                    rows.new_full((), rows.numel(), dtype=torch.int64),
                )
            ).tolist()
        )
        if minimum_row < 0 or maximum_row >= self.num_envs:
            raise ValueError(
                "reset rows are outside the environment range "
                f"[0, {self.num_envs}); got [{minimum_row}, {maximum_row}]"
            )
        if unique_count != validated_count:
            raise ValueError(
                f"reset rows must be unique; got {validated_count - unique_count} duplicate row(s)"
            )
        return rows

    def reset(self, rows: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        selected = (
            torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
            if rows is None
            else self._validate_rows(rows)
        )
        if selected.numel():
            result = self.backend.set_state_tensor(
                selected,
                self.reset_qpos.index_select(0, selected),
                self.reset_qvel.index_select(0, selected),
            )
            if result is not None and not isinstance(result, dict):
                raise TypeError("backend tensor reset must return a dict or None")
            self._raw_actions[selected] = 0.0
            self._previous_raw_actions[selected] = 0.0
            self._steps[selected] = 0
            self._phase[selected] = 0.0
        self._update_observation()
        return {name: values.clone() for name, values in self._obs.items()}

    def step(self, actions: torch.Tensor, nsteps: int = 2) -> Go2TensorParityState:
        actions = self._validate_actions(actions)
        if isinstance(nsteps, bool) or int(nsteps) != nsteps or int(nsteps) <= 0:
            raise ValueError("nsteps must be a positive integer")
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        self._ctrl.copy_(self._default_ctrl)
        self._ctrl.add_(self._raw_actions, alpha=self.cfg.action_scale)
        result = self.backend.step_tensor(self._ctrl, int(nsteps))
        if result is not None and not isinstance(result, dict):
            raise TypeError("backend tensor step must return a dict or None")

        self._steps += 1
        self._phase = torch.remainder(
            self._phase + self.cfg.ctrl_dt * self.cfg.gait_frequency,
            1.0,
        )
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        self._update_observation()
        self._compute_rewards()

        reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        weights = {
            "tracking_lin_vel": self.cfg.tracking_lin_vel_weight,
            "tracking_ang_vel": self.cfg.tracking_ang_vel_weight,
            "lin_vel_z": self.cfg.lin_vel_z_weight,
            "ang_vel_xy": self.cfg.ang_vel_xy_weight,
            "base_height": self.cfg.base_height_weight,
            "action_rate": self.cfg.action_rate_weight,
            "similar_to_default": self.cfg.joint_deviation_weight,
            "contact": self.cfg.contact_weight,
            "swing_feet_z": self.cfg.swing_feet_weight,
        }
        for name, weight in weights.items():
            reward.add_(self.reward_components[name], alpha=weight * self.cfg.ctrl_dt)
        return Go2TensorParityState(
            obs=self._obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"steps": self._steps.clone()},
        )

    def _compute_terminated(self) -> torch.Tensor:
        upvector_z = self.sensors["upvector"][:, 2].clamp(-1.0, 1.0)
        angle = torch.arccos(upvector_z)
        return angle > self.cfg.bad_orientation_limit

    def _compute_truncated(self) -> torch.Tensor:
        return self._steps >= self.cfg.max_episode_steps

    def _update_observation(self) -> None:
        joint_pos = self._qpos[:, 7:]
        joint_vel = self._qvel[:, 6:]
        phase = torch.remainder(self._phase[:, None] + self._gait_offsets, 1.0)
        policy_parts = (
            self.sensors["gyro"],
            -self.sensors["upvector"],
            joint_pos - self._default_joint_pos,
            joint_vel - self._default_joint_vel,
            self._raw_actions,
            self.command,
            phase,
        )
        policy = torch.cat(policy_parts, dim=1)
        if policy.shape != (self.num_envs, 49):
            raise RuntimeError(f"Go2 policy observation width is {policy.shape[1]}, expected 49")
        self._obs["policy"].copy_(policy)
        self._obs["critic"].copy_(torch.cat((policy, self.sensors["local_linvel"]), dim=1))

    def _compute_rewards(self) -> None:
        lin_vel = self.sensors["local_linvel"]
        gyro = self.sensors["gyro"]
        joint_pos = self._qpos[:, 7:]
        foot_heights = torch.stack(
            [self.sensors[name][:, 2] for name in _FOOT_POSITION_SENSOR_NAMES], dim=1
        )
        contacts = torch.cat([self.sensors[name] for name in _FOOT_CONTACT_SENSOR_NAMES], dim=1)
        phases = torch.remainder(self._phase[:, None] + self._gait_offsets, 1.0)

        lin_error = torch.sum(torch.square(self.command[:, :2] - lin_vel[:, :2]), dim=1)
        torch.exp(
            -lin_error / self.cfg.tracking_lin_vel_std**2,
            out=self.reward_components["tracking_lin_vel"],
        )
        ang_error = torch.square(self.command[:, 2] - gyro[:, 2])
        torch.exp(
            -ang_error / self.cfg.tracking_ang_vel_std**2,
            out=self.reward_components["tracking_ang_vel"],
        )
        torch.square(lin_vel[:, 2], out=self.reward_components["lin_vel_z"])
        torch.sum(torch.square(gyro[:, :2]), dim=1, out=self.reward_components["ang_vel_xy"])
        torch.square(
            self._qpos[:, 2] - self.cfg.base_height_target,
            out=self.reward_components["base_height"],
        )
        torch.sum(
            torch.square(self._raw_actions - self._previous_raw_actions),
            dim=1,
            out=self.reward_components["action_rate"],
        )
        torch.sum(
            torch.abs(joint_pos - self._default_joint_pos),
            dim=1,
            out=self.reward_components["similar_to_default"],
        )
        contact_active = contacts > self.cfg.contact_threshold
        expected_contact = phases < self.cfg.stance_threshold
        torch.mean(
            (contact_active == expected_contact).to(torch.float32),
            dim=1,
            out=self.reward_components["contact"],
        )
        swing = phases >= self.cfg.swing_start
        height_reward = torch.exp(
            -torch.square(foot_heights - self.cfg.foot_target_height) / self.cfg.foot_height_kernel
        )
        torch.mean(
            height_reward * swing,
            dim=1,
            out=self.reward_components["swing_feet_z"],
        )


__all__ = [
    "GO2_ACTUATOR_NAMES",
    "GO2_JOINT_NAMES",
    "Go2TensorParityConfig",
    "Go2TensorParityRuntime",
    "Go2TensorParityState",
]
