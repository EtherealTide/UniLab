"""Manager-Based terms for Stewart-platform ball balancing.

Hydra owns the production task declaration.  This module contains only the
task-specific NumPy terms and the generic Manager-Based registry binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env
from unilab.managers import ActionTerm, ActionTermCfg, ManagerTermBase, ManagerTermBaseCfg
from unilab.tasks.motion_tracking.common.tensor_rotation import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_roll_pitch,
    quat_to_axis_angle,
)
from unilab.utils.rotation import (
    np_quat_apply_inverse,
)

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.termination_manager import TerminationManager

    class _StewartEnv(ManagerBasedRlEnv, Protocol):
        @property
        def common_step_counter(self) -> int: ...

        observation_manager: ObservationManager


_ACTION_DIM = 2
_LEG_COUNT = 6


def _real(
    term: str,
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{term} {name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{term} {name} must be finite")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{term} {name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{term} {name} must be at most {maximum}")
    return result


def _name(term: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{term} {name} must be a non-empty string")
    return value


def _names(term: str, name: str, value: Any, *, count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a sequence of {count} strings")
    result = tuple(value)
    if len(result) != count:
        raise ValueError(f"{term} {name} must contain {count} names, got {len(result)}")
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{term} {name} must contain non-empty strings")
    if len(set(result)) != count:
        raise ValueError(f"{term} {name} must contain unique names: {result}")
    return result


def _pair(term: str, name: str, value: Any) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{term} {name} must be a numeric (min, max) pair")
    if len(value) != 2:
        raise ValueError(f"{term} {name} must contain two values")
    lower = _real(term, f"{name}[0]", value[0])
    upper = _real(term, f"{name}[1]", value[1])
    if lower > upper:
        raise ValueError(f"{term} {name} lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _env_ids(env: ManagerBasedRlEnv, env_ids: np.ndarray | slice | None) -> np.ndarray:
    if env_ids is None:
        return np.arange(env.num_envs, dtype=np.int32)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs, dtype=np.int32)[env_ids]
    return env_ids


def _body_id(entity: Entity, name: str, *, term: str) -> int:
    ids, resolved = entity.find_bodies(name)
    if len(ids) != 1 or resolved != [name]:
        raise ValueError(f"{term} body selector {name!r} did not resolve exactly once")
    return ids[0]


def _body_ids(entity: Entity, names: tuple[str, ...], *, term: str) -> np.ndarray:
    ids, resolved = entity.find_bodies(names, preserve_order=True)
    if tuple(resolved) != names:
        raise ValueError(f"{term} body selectors resolved in an unexpected order: {resolved}")
    result = np.asarray(ids, dtype=np.intp)
    result.setflags(write=False)
    return result


def _relative_ball_state(
    entity: Entity,
    *,
    ball_body_id: int,
    top_body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_pos = entity.data.body_link_pos_w
    body_quat = entity.data.body_link_quat_w
    top_pos = body_pos[:, top_body_id]
    top_quat = body_quat[:, top_body_id]
    ball_pos = body_pos[:, ball_body_id]
    relative = np_quat_apply_inverse(top_quat, ball_pos - top_pos)
    return relative, top_quat, ball_pos


def _runtime_tensor(values: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    """Return float32 Torch state for the task's Torch manager buffers."""
    if isinstance(values, torch.Tensor):
        if values.dtype != torch.float32:
            raise TypeError(f"StewartBalanceState {name} must be float32 Torch values")
        return values
    return torch.tensor(np.asarray(values), dtype=torch.float32)


@dataclass(kw_only=True)
class StewartTiltActionCfg(ActionTermCfg):
    """Two-axis tilt action converted to six Stewart actuator targets."""

    actuator_names: tuple[str, ...] | list[str]
    top_body_name: str
    ball_body_name: str
    leg_body_names: tuple[str, ...] | list[str]
    top_connect_body_names: tuple[str, ...] | list[str]
    raw_action_clip: tuple[float, float] | list[float]
    target_rotation_limit_deg: float
    action_smooth: float
    center_control_radius: float
    center_control_min_gain: float

    def build(self, env: ManagerBasedRlEnv) -> StewartTiltAction:
        return StewartTiltAction(self, env)


class StewartTiltAction(ActionTerm):
    """Vectorized tilt IK using only the public entity state/control facade."""

    cfg: StewartTiltActionCfg
    _entity: Entity
    uses_tensor_actions = True

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (*self._leg_names, *self._connect_names, self._top_name, self._ball_name)

    def __init__(self, cfg: StewartTiltActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        term = type(self).__name__
        if cfg.clip is not None:
            raise NotImplementedError(
                f"{term} does not support the actuator-name clip field; use raw_action_clip"
            )
        actuator_names = _names(term, "actuator_names", cfg.actuator_names, count=_LEG_COUNT)
        actuator_ids, resolved = self._entity.find_actuators(actuator_names, preserve_order=True)
        if tuple(resolved) != actuator_names:
            raise ValueError(f"{term} actuator selectors resolved out of order: {resolved}")
        self._actuator_ids = np.asarray(actuator_ids, dtype=np.intp)
        self._actuator_ids.setflags(write=False)

        top_name = _name(term, "top_body_name", cfg.top_body_name)
        ball_name = _name(term, "ball_body_name", cfg.ball_body_name)
        leg_names = _names(term, "leg_body_names", cfg.leg_body_names, count=_LEG_COUNT)
        connect_names = _names(
            term,
            "top_connect_body_names",
            cfg.top_connect_body_names,
            count=_LEG_COUNT,
        )
        self._top_body_id = _body_id(self._entity, top_name, term=term)
        self._ball_body_id = _body_id(self._entity, ball_name, term=term)
        self._leg_body_ids = _body_ids(self._entity, leg_names, term=term)
        self._top_connect_body_ids = _body_ids(self._entity, connect_names, term=term)
        self._top_name = top_name
        self._ball_name = ball_name
        self._leg_names = leg_names
        self._connect_names = connect_names

        self._raw_clip = _pair(term, "raw_action_clip", cfg.raw_action_clip)
        self._tilt_limit_deg = _real(
            term,
            "target_rotation_limit_deg",
            cfg.target_rotation_limit_deg,
            minimum=0.0,
            strict_minimum=True,
        )
        self._action_smooth = _real(
            term, "action_smooth", cfg.action_smooth, minimum=0.0, maximum=1.0
        )
        self._center_radius = _real(
            term, "center_control_radius", cfg.center_control_radius, minimum=0.0
        )
        self._center_min_gain = _real(
            term,
            "center_control_min_gain",
            cfg.center_control_min_gain,
            minimum=0.0,
            maximum=1.0,
        )

        ranges = np.asarray(self._entity.data.actuator_ctrl_range, dtype=get_global_dtype())
        self._device = torch.device(getattr(env, "device", "cpu"))
        self._ctrl_lower = torch.from_numpy(
            np.ascontiguousarray(ranges[self._actuator_ids, 0], dtype=np.float32)
        ).to(self._device)
        self._ctrl_upper = torch.from_numpy(
            np.ascontiguousarray(ranges[self._actuator_ids, 1], dtype=np.float32)
        ).to(self._device)
        self._raw_action = torch.zeros(
            (env.num_envs, _ACTION_DIM), dtype=torch.float32, device=self._device
        )
        self._clipped_action = torch.zeros_like(self._raw_action)
        self._executed_action = torch.zeros_like(self._raw_action)
        self._previous_executed_action = torch.zeros_like(self._raw_action)
        self._effective_action = torch.zeros_like(self._raw_action)
        self._target_tilt_deg = torch.zeros_like(self._raw_action)
        self._target_tilt_rad = torch.zeros_like(self._raw_action)
        self._control = torch.zeros(
            (env.num_envs, _LEG_COUNT), dtype=torch.float32, device=self._device
        )

        self._ik_ready = False
        self._top_home_pos = torch.zeros(3, dtype=torch.float32, device=self._device)
        self._connect_offsets = torch.zeros(
            (_LEG_COUNT, 3), dtype=torch.float32, device=self._device
        )
        self._neutral_leg_lengths = torch.zeros(
            _LEG_COUNT, dtype=torch.float32, device=self._device
        )

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def executed_action(self) -> torch.Tensor:
        return self._executed_action

    @property
    def target_tilt_deg(self) -> torch.Tensor:
        return self._target_tilt_deg

    @property
    def neutral_leg_lengths(self) -> torch.Tensor:
        self._ensure_ik_calibration()
        return self._neutral_leg_lengths

    def _body_state(self, body_names: tuple[str, ...]):
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            if not read_plan.ready:
                read_plan.refresh()
            return read_plan.body_tensor_view(self._entity, body_names)
        return self._entity.body_tensor_view(self._device, body_names=body_names)

    def _ensure_ik_calibration(self) -> None:
        if self._ik_ready:
            return
        state = self._body_state((*self._leg_names, *self._connect_names, self._top_name))
        positions = dict(zip(state.body_names, state.pos_w.unbind(dim=1), strict=True))
        first = {name: value[0] for name, value in positions.items()}
        top = first[self._top_name]
        connects = torch.stack(tuple(first[name] for name in self._connect_names), dim=0)
        legs = torch.stack(tuple(first[name] for name in self._leg_names), dim=0)
        self._top_home_pos.copy_(top)
        self._connect_offsets.copy_(connects - top)
        self._neutral_leg_lengths.copy_(torch.linalg.vector_norm(connects - legs, dim=-1))
        self._ik_ready = True

    def _relative_ball_state(self) -> torch.Tensor:
        state = self._body_state((self._top_name, self._ball_name))
        positions = dict(zip(state.body_names, state.pos_w.unbind(dim=1), strict=True))
        orientations = dict(zip(state.body_names, state.quat_w.unbind(dim=1), strict=True))
        top_pos = positions[self._top_name]
        top_quat = orientations[self._top_name]
        ball_pos = positions[self._ball_name]
        return quat_apply_inverse(top_quat, ball_pos - top_pos)

    def leg_control_for_tilt(self, target_tilt_rad: torch.Tensor) -> torch.Tensor:
        """Return six actuator controls for ``(roll, pitch)`` radians."""
        expected = (self.num_envs, _ACTION_DIM)
        if not isinstance(target_tilt_rad, torch.Tensor) or target_tilt_rad.shape != expected:
            shape = getattr(target_tilt_rad, "shape", None)
            raise ValueError(f"{type(self).__name__} tilt must have shape {expected}, got {shape}")
        if target_tilt_rad.dtype != torch.float32 or target_tilt_rad.device != self._device:
            raise TypeError(
                f"{type(self).__name__} tilt must be float32 on {self._device}; got "
                f"{target_tilt_rad.dtype} on {target_tilt_rad.device}"
            )
        if not bool(torch.isfinite(target_tilt_rad).all()):
            raise ValueError(f"{type(self).__name__} tilt contains NaN or Inf")
        self._ensure_ik_calibration()
        zeros = torch.zeros_like(target_tilt_rad[:, 0])
        target_quat = quat_from_euler_xyz(target_tilt_rad[:, 0], target_tilt_rad[:, 1], zeros)
        rotated = quat_apply(target_quat[:, None, :], self._connect_offsets[None, :, :])
        expected_connects = self._top_home_pos + rotated
        leg_positions = self._body_state(self._leg_names).pos_w
        controls = (
            torch.linalg.vector_norm(expected_connects - leg_positions, dim=-1)
            - self._neutral_leg_lengths
        )
        return torch.clamp(controls, min=self._ctrl_lower, max=self._ctrl_upper)

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = self._raw_action.shape
        if not isinstance(actions, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__} expected torch.Tensor, got {type(actions).__name__}"
            )
        if actions.shape != expected:
            raise ValueError(
                f"{type(self).__name__} expected action shape {expected}, got {actions.shape}"
            )
        if actions.dtype != torch.float32 or actions.device != self._device:
            raise TypeError(
                f"{type(self).__name__} expected float32 actions on {self._device}; got "
                f"{actions.dtype} on {actions.device}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise ValueError(f"{type(self).__name__} received NaN or Inf actions")
        self._raw_action.copy_(actions)
        torch.clamp(actions, min=self._raw_clip[0], max=self._raw_clip[1], out=self._clipped_action)
        torch.mul(self._clipped_action, self._action_smooth, out=self._executed_action)
        self._executed_action.add_((1.0 - self._action_smooth) * self._previous_executed_action)
        self._previous_executed_action.copy_(self._executed_action)

        relative = self._relative_ball_state()
        relative_xy = torch.linalg.vector_norm(relative[:, :2], dim=-1)
        if self._center_radius > 0.0 and self._center_min_gain < 1.0:
            ratio = torch.clamp(relative_xy / self._center_radius, 0.0, 1.0)
            gain = self._center_min_gain + (1.0 - self._center_min_gain) * ratio
            torch.mul(self._executed_action, gain[:, None], out=self._effective_action)
        else:
            self._effective_action.copy_(self._executed_action)
        torch.mul(self._effective_action, self._tilt_limit_deg, out=self._target_tilt_deg)
        torch.deg2rad(self._target_tilt_deg, out=self._target_tilt_rad)
        self._control.copy_(self.leg_control_for_tilt(self._target_tilt_rad))

    def apply_actions(self) -> None:
        self._entity.data.write_ctrl(self._control, actuator_ids=self._actuator_ids)

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self._raw_action,
            self._clipped_action,
            self._executed_action,
            self._previous_executed_action,
            self._effective_action,
            self._target_tilt_deg,
            self._target_tilt_rad,
            self._control,
        ):
            value[ids] = 0.0


class StewartObservation(ManagerTermBase):
    """Legacy 15-D observation with per-environment filtered finite differences."""

    _ALLOWED_PARAMS = frozenset(
        {
            "entity_name",
            "action_name",
            "ball_body_name",
            "top_body_name",
            "target_rotation_limit_deg",
            "vel_smooth",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _StewartEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        action_name = _name(term, "action_name", cfg.params.get("action_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        self._ball_body_id = _body_id(
            self._entity,
            _name(term, "ball_body_name", cfg.params.get("ball_body_name")),
            term=term,
        )
        self._top_body_id = _body_id(
            self._entity,
            _name(term, "top_body_name", cfg.params.get("top_body_name")),
            term=term,
        )
        action = env.action_manager.get_term(action_name)
        if not isinstance(action, StewartTiltAction):
            raise TypeError(
                f"{term} action term {action_name!r} must be StewartTiltAction, "
                f"got {type(action).__name__}"
            )
        self._action = action
        self._top_name = _name(term, "top_body_name", cfg.params.get("top_body_name"))
        self._ball_name = _name(term, "ball_body_name", cfg.params.get("ball_body_name"))
        self._tilt_limit_deg = _real(
            term,
            "target_rotation_limit_deg",
            cfg.params.get("target_rotation_limit_deg"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._vel_smooth = _real(
            term,
            "vel_smooth",
            cfg.params.get("vel_smooth"),
            minimum=0.0,
            maximum=1.0,
        )
        self._step_dt = _real(term, "step_dt", env.step_dt, minimum=0.0, strict_minimum=True)

        self._device = torch.device(getattr(env, "device", "cpu"))
        self._relative = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self._device)
        self._previous_relative = torch.zeros_like(self._relative)
        self._filtered_relative_velocity = torch.zeros_like(self._relative)
        self._top_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=self._device)
        self._top_quat[:, 0] = 1.0
        self._previous_top_quat = self._top_quat.clone()
        self._filtered_top_angular_velocity = torch.zeros_like(self._relative)
        self._local_top_angular_velocity = torch.zeros_like(self._relative)
        self._ball_pos = torch.zeros_like(self._relative)
        self._relative_xy = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self._velocity_xy = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self._obs = torch.zeros((env.num_envs, 15), dtype=torch.float32, device=self._device)
        self._last_counter = self._counter(env)

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return (self._top_name, self._ball_name)

    @staticmethod
    def _counter(env: _StewartEnv) -> int:
        counter = env.common_step_counter
        if isinstance(counter, (bool, np.bool_)) or not isinstance(counter, (int, np.integer)):
            raise TypeError("StewartObservation common_step_counter must be an integer")
        if counter < 0:
            raise ValueError("StewartObservation common_step_counter must be non-negative")
        return int(counter)

    @property
    def relative_xy(self) -> torch.Tensor:
        return self._relative_xy

    @property
    def velocity_xy(self) -> torch.Tensor:
        return self._velocity_xy

    @property
    def ball_pos(self) -> torch.Tensor:
        return self._ball_pos

    def _write_observation_rows(self, ids: np.ndarray, *, reset_actions: bool) -> None:
        roll, pitch = quat_roll_pitch(self._top_quat[ids])
        self._obs[ids, 0:3] = self._relative[ids]
        self._obs[ids, 3:6] = self._filtered_relative_velocity[ids]
        self._obs[ids, 6] = torch.rad2deg(roll) / self._tilt_limit_deg
        self._obs[ids, 7] = torch.rad2deg(pitch) / self._tilt_limit_deg
        self._obs[ids, 8:11] = self._local_top_angular_velocity[ids]
        if reset_actions:
            self._obs[ids, 11:15] = 0.0
        else:
            self._obs[ids, 11:13] = self._action.target_tilt_deg[ids] / self._tilt_limit_deg
            self._obs[ids, 13:15] = self._action.executed_action[ids]

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is not None and not read_plan.ready:
            # Initial manager construction runs before the first reset-state
            # transaction publishes a packet. The Manager-owned reset below is
            # the authoritative row-scoped initialization.
            return
        relative, top_quat, ball_pos = self._relative_ball_state()
        self._relative[ids] = relative[ids]
        self._previous_relative[ids] = relative[ids]
        self._filtered_relative_velocity[ids] = 0.0
        self._top_quat[ids] = top_quat[ids]
        self._previous_top_quat[ids] = top_quat[ids]
        self._filtered_top_angular_velocity[ids] = 0.0
        self._local_top_angular_velocity[ids] = 0.0
        self._ball_pos[ids] = ball_pos[ids]
        self._relative_xy[ids] = torch.linalg.vector_norm(relative[ids, :2], dim=-1)
        self._velocity_xy[ids] = 0.0
        self._write_observation_rows(ids, reset_actions=True)

    def _advance(self, env: _StewartEnv) -> None:
        counter = self._counter(env)
        if counter == self._last_counter:
            return
        if counter != self._last_counter + 1:
            raise RuntimeError(
                "StewartObservation missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )
        relative, top_quat, ball_pos = self._relative_ball_state()
        relative_velocity = (relative - self._previous_relative) / self._step_dt
        self._filtered_relative_velocity[:] = (
            self._vel_smooth * relative_velocity
            + (1.0 - self._vel_smooth) * self._filtered_relative_velocity
        )
        quaternion_delta = quat_mul(top_quat, quat_conjugate(self._previous_top_quat))
        top_angular_velocity = quat_to_axis_angle(quaternion_delta) / self._step_dt
        self._filtered_top_angular_velocity[:] = (
            self._vel_smooth * top_angular_velocity
            + (1.0 - self._vel_smooth) * self._filtered_top_angular_velocity
        )
        self._local_top_angular_velocity[:] = quat_apply_inverse(
            top_quat, self._filtered_top_angular_velocity
        )
        self._relative[:] = relative
        self._previous_relative[:] = relative
        self._top_quat[:] = top_quat
        self._previous_top_quat[:] = top_quat
        self._ball_pos[:] = ball_pos
        self._relative_xy[:] = torch.linalg.vector_norm(relative[:, :2], dim=-1)
        self._velocity_xy[:] = torch.linalg.vector_norm(
            self._filtered_relative_velocity[:, :2], dim=-1
        )
        all_ids = np.arange(env.num_envs, dtype=np.int32)
        self._write_observation_rows(all_ids, reset_actions=False)
        self._last_counter = counter

    def _relative_ball_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        read_plan = getattr(self._env.scene, "_tensor_read_plan", None)
        if read_plan is None:
            raise NotImplementedError(
                "StewartObservation requires a scene-owned packed tensor read plan; "
                "compile tensor body reads with the Manager-Based Torch runtime"
            )
        state = read_plan.body_tensor_view(self._entity, (self._top_name, self._ball_name))
        positions = dict(zip(state.body_names, state.pos_w.unbind(dim=1), strict=True))
        orientations = dict(zip(state.body_names, state.quat_w.unbind(dim=1), strict=True))
        top_pos = positions[self._top_name]
        top_quat = orientations[self._top_name]
        ball_pos = positions[self._ball_name]
        relative = quat_apply_inverse(top_quat, ball_pos - top_pos)
        return relative, top_quat, ball_pos

    def snapshot(self, env: _StewartEnv) -> torch.Tensor:
        self._advance(env)
        return self._obs

    def __call__(self, env: _StewartEnv, **params: Any) -> torch.Tensor:
        del params
        return self.snapshot(env)


class StewartBalanceState(ManagerTermBase):
    """Termination-owned progress, stillness, success, and fall state."""

    _ALLOWED_PARAMS = frozenset(
        {
            "observation_group",
            "observation_term",
            "platform_radius",
            "fall_radius",
            "top_center_z",
            "still_xy",
            "still_vel",
            "still_xy_hysteresis",
            "still_vel_hysteresis",
            "zero_vel_thresh",
            "still_steps_needed",
        }
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: _StewartEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        group_name = _name(term, "observation_group", cfg.params.get("observation_group"))
        observation_name = _name(term, "observation_term", cfg.params.get("observation_term"))
        observation = env.observation_manager.get_term_cfg(group_name, observation_name).func
        if not isinstance(observation, StewartObservation):
            raise TypeError(
                f"{term} observation {group_name}/{observation_name} must be "
                f"StewartObservation, got {type(observation).__name__}"
            )
        self._observation = observation
        self._platform_radius = _real(
            term,
            "platform_radius",
            cfg.params.get("platform_radius"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._fall_radius = _real(
            term,
            "fall_radius",
            cfg.params.get("fall_radius"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._top_center_z = _real(term, "top_center_z", cfg.params.get("top_center_z"))
        self._still_xy = _real(term, "still_xy", cfg.params.get("still_xy"), minimum=0.0)
        self._still_vel = _real(term, "still_vel", cfg.params.get("still_vel"), minimum=0.0)
        self._still_xy_hysteresis = _real(
            term,
            "still_xy_hysteresis",
            cfg.params.get("still_xy_hysteresis"),
            minimum=1.0,
        )
        self._still_vel_hysteresis = _real(
            term,
            "still_vel_hysteresis",
            cfg.params.get("still_vel_hysteresis"),
            minimum=1.0,
        )
        self._zero_vel_thresh = _real(
            term,
            "zero_vel_thresh",
            cfg.params.get("zero_vel_thresh"),
            minimum=0.0,
        )
        steps = cfg.params.get("still_steps_needed")
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
            raise TypeError(f"{term} still_steps_needed must be an integer")
        if steps <= 0:
            raise ValueError(f"{term} still_steps_needed must be positive")
        self._still_steps_needed = int(steps)

        self._device = torch.device(getattr(env, "device", "cpu"))
        self.fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self.center_score = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self.progress = torch.zeros(env.num_envs, dtype=torch.float32, device=self._device)
        self.still_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=self._device)
        self.still_window_active = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self._previous_zero_velocity_xy = torch.zeros(
            env.num_envs, dtype=torch.float32, device=self._device
        )
        self._done = torch.zeros(env.num_envs, dtype=torch.bool, device=self._device)
        self._last_counter = int(env.common_step_counter)

    @property
    def last_counter(self) -> int:
        return self._last_counter

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        ids = _env_ids(self._env, env_ids)
        self.fallen[ids] = False
        self.success[ids] = False
        self.center_score[ids] = torch.clamp(
            1.0 - self._observation.relative_xy[ids] / self._fall_radius, min=0.0, max=1.0
        )
        self.progress[ids] = 0.0
        self.still_steps[ids] = 0
        self.still_window_active[ids] = False
        self._previous_zero_velocity_xy[ids] = self._observation.relative_xy[ids]
        self._done[ids] = False

    def _update(
        self,
        relative_xy: torch.Tensor | np.ndarray,
        velocity_xy: torch.Tensor | np.ndarray,
        ball_pos: torch.Tensor | np.ndarray,
    ) -> None:
        relative_xy = _runtime_tensor(relative_xy, name="relative_xy")
        velocity_xy = _runtime_tensor(velocity_xy, name="velocity_xy")
        ball_pos = _runtime_tensor(ball_pos, name="ball_pos")
        fall_z = self._top_center_z - np.sin(np.deg2rad(30.0)) * self._platform_radius
        self.fallen[:] = (relative_xy > self._fall_radius) | (ball_pos[:, 2] < fall_z)
        self.center_score[:] = torch.clip(
            1.0 - relative_xy / self._fall_radius,
            0.0,
            1.0,
        )

        zero_event = velocity_xy <= self._zero_vel_thresh
        improvement = torch.clamp(self._previous_zero_velocity_xy - relative_xy, min=0.0)
        self.progress[:] = torch.where(
            zero_event & (relative_xy < self._previous_zero_velocity_xy),
            torch.clip(improvement / self._platform_radius, 0.0, 1.0),
            torch.zeros_like(improvement),
        )
        self._previous_zero_velocity_xy[zero_event] = relative_xy[zero_event]

        keep = (
            self.still_window_active
            & (relative_xy <= self._still_xy * self._still_xy_hysteresis)
            & (velocity_xy <= self._still_vel * self._still_vel_hysteresis)
        )
        enter = (
            ~self.still_window_active
            & (relative_xy <= self._still_xy)
            & (velocity_xy <= self._still_vel)
        )
        self.still_steps[:] = torch.where(
            keep,
            self.still_steps + 1,
            torch.where(
                enter, torch.ones_like(self.still_steps), torch.zeros_like(self.still_steps)
            ),
        )
        self.still_window_active[:] = keep | enter
        self.success[:] = self.still_steps >= self._still_steps_needed
        self._done[:] = self.fallen | self.success

    def __call__(self, env: _StewartEnv, **params: Any) -> torch.Tensor:
        del params
        counter = int(env.common_step_counter)
        if counter == self._last_counter:
            return self._done
        if counter != self._last_counter + 1:
            raise RuntimeError(
                "StewartBalanceState missed a control-step update: "
                f"last={self._last_counter}, current={counter}"
            )
        self._observation.snapshot(env)
        self._update(
            self._observation.relative_xy,
            self._observation.velocity_xy,
            self._observation.ball_pos,
        )
        self._last_counter = counter
        return self._done


def _balance_state(env: _StewartEnv, state_term_name: str) -> StewartBalanceState:
    name = _name("Stewart reward", "state_term_name", state_term_name)
    termination_manager = cast("TerminationManager", env.termination_manager)
    state = termination_manager.get_term_cfg(name).func
    if not isinstance(state, StewartBalanceState):
        raise TypeError(
            f"Stewart reward termination term {name!r} must be StewartBalanceState, "
            f"got {type(state).__name__}"
        )
    if state.last_counter != int(env.common_step_counter):
        raise RuntimeError(
            f"Stewart reward state {name!r} was not computed for control step "
            f"{env.common_step_counter}"
        )
    return state


def center_reward(env: _StewartEnv, state_term_name: str) -> np.ndarray:
    state = _balance_state(env, state_term_name)
    return np.asarray(
        torch.where(
            state.fallen,
            torch.zeros_like(state.center_score),
            state.center_score,
        )
        .cpu()
        .numpy(),
        dtype=get_global_dtype(),
    )


def progress_reward(env: _StewartEnv, state_term_name: str) -> np.ndarray:
    state = _balance_state(env, state_term_name)
    return np.asarray(
        torch.where(
            state.fallen,
            torch.zeros_like(state.progress),
            state.progress,
        )
        .cpu()
        .numpy(),
        dtype=get_global_dtype(),
    )


def still_reward(env: _StewartEnv, state_term_name: str) -> np.ndarray:
    state = _balance_state(env, state_term_name)
    return np.asarray((state.success & ~state.fallen).cpu().numpy(), dtype=get_global_dtype())


def fall_reward(env: _StewartEnv, state_term_name: str) -> np.ndarray:
    state = _balance_state(env, state_term_name)
    return np.asarray(state.fallen.cpu().numpy(), dtype=get_global_dtype())


class StewartBallReset(ManagerTermBase):
    """Sample the ball uniformly within a disk via root-state entity writes."""

    _ALLOWED_PARAMS = frozenset(
        {"entity_name", "platform_radius", "init_ball_radius_ratio", "ball_home_z"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        term = type(self).__name__
        unexpected = set(cfg.params) - self._ALLOWED_PARAMS
        if unexpected:
            raise TypeError(f"{term} received unsupported parameters: {sorted(unexpected)}")
        entity_name = _name(term, "entity_name", cfg.params.get("entity_name"))
        self._entity = cast("Entity", env.scene[entity_name])
        self._platform_radius = _real(
            term,
            "platform_radius",
            cfg.params.get("platform_radius"),
            minimum=0.0,
            strict_minimum=True,
        )
        self._radius_ratio = _real(
            term,
            "init_ball_radius_ratio",
            cfg.params.get("init_ball_radius_ratio"),
            minimum=0.0,
            maximum=1.0,
        )
        self._ball_home_z = _real(term, "ball_home_z", cfg.params.get("ball_home_z"))
        # Resolve the complete floating-root capability on the cold path.
        default_state = self._entity.data.default_root_state
        if default_state.shape != (env.num_envs, 13):
            raise ValueError(
                f"{term} default root state must have shape ({env.num_envs}, 13), "
                f"got {default_state.shape}"
            )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | None,
        **params: Any,
    ) -> None:
        del params
        ids = _env_ids(env, env_ids)
        root_state = np.array(self._entity.data.default_root_state[ids], copy=True)
        radius = (
            self._platform_radius
            * self._radius_ratio
            * np.sqrt(env.rng.uniform(0.0, 1.0, size=ids.size))
        )
        theta = env.rng.uniform(0.0, 2.0 * np.pi, size=ids.size)
        root_state[:, 0] = radius * np.cos(theta)
        root_state[:, 1] = radius * np.sin(theta)
        root_state[:, 2] = self._ball_home_z
        self._entity.write_root_link_pose_to_sim(root_state[:, :7], env_ids=ids)
        self._entity.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=ids)


registry.register_env_config("StewartBalance", ManagerBasedRlEnvCfg)
registry.register_env("StewartBalance", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("StewartBalance", make_manager_based_rl_env, sim_backend="motrix")
registry.register_env("StewartBalance", make_manager_based_rl_env, sim_backend="drake")


__all__ = [
    "StewartBalanceState",
    "StewartBallReset",
    "StewartObservation",
    "StewartTiltAction",
    "StewartTiltActionCfg",
    "center_reward",
    "fall_reward",
    "progress_reward",
    "still_reward",
]
