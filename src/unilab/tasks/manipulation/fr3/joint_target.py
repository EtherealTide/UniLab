"""Joint-target terms using the public entity/reset contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers import ManagerTermBaseCfg
    from unilab.managers._types import ManagerBasedRlEnv


class JointTargetObservation:
    """Bind one ordered joint target on the manager construction path."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        self._entity = cast("Entity", env.scene[cfg.params["entity_name"]])
        self.tensor_state_entity = cast(str, cfg.params["entity_name"])
        self._target = torch.as_tensor(
            list(cfg.params["target"]), dtype=torch.float32, device=env.device
        )
        expected = (self._entity.num_joints,)
        if self._target.shape != expected or not bool(torch.isfinite(self._target).all()):
            raise ValueError(f"FR3 joint target must be finite with shape {expected}")
        self._target = self._target.contiguous()

    def __call__(
        self, env: ManagerBasedRlEnv, entity_name: str, target: list[float]
    ) -> np.ndarray | torch.Tensor:
        del entity_name, target
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            return read_plan.joint_tensor_view(self._entity).joint_pos - self._target
        return self._entity.data.joint_pos - self._target.numpy()


class JointTargetReward(JointTargetObservation):
    """Reward current joint accuracy with a configured squared-error scale."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        std = cfg.params["std"]
        if (
            isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not torch.isfinite(torch.tensor(float(std)))
            or std <= 0
        ):
            raise ValueError("FR3 joint target reward std must be finite and positive")
        self._variance = float(std) ** 2

    def __call__(
        self, env: ManagerBasedRlEnv, entity_name: str, target: list[float], std: float = 0.5
    ) -> np.ndarray | torch.Tensor:
        read_plan = getattr(env.scene, "_tensor_read_plan", None)
        if read_plan is not None:
            error: np.ndarray | torch.Tensor = (
                read_plan.joint_tensor_view(self._entity).joint_pos - self._target
            )
        else:
            error = (
                torch.as_tensor(
                    self._entity.data.joint_pos, dtype=torch.float32, device=self._target.device
                )
                - self._target
            )
        if isinstance(error, torch.Tensor):
            return torch.exp(-torch.sum(torch.square(error), dim=-1) / self._variance)
        return torch.exp(-torch.sum(torch.square(torch.as_tensor(error)), dim=-1) / self._variance)


class ResetJointOffsets:
    """Stage selected joint resets without requiring a floating root."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        self._entity = cast("Entity", env.scene[cfg.params["entity_name"]])
        self._ranges = []
        self.tensor_state_entity = cast(str, cfg.params["entity_name"])
        self.uses_tensor_rows = True
        for name in ("position_range", "velocity_range"):
            values = np.asarray(cfg.params[name], dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"FR3 reset {name} must be a finite ordered pair")
            self._ranges.append((float(values[0]), float(values[1])))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        entity_name: str,
        position_range: list[float],
        velocity_range: list[float],
    ) -> None:
        if env_ids is None:
            raise ValueError("FR3 reset requires explicit environment IDs")
        positions = self._entity.data.default_joint_pos_torch(env.device).index_select(0, env_ids)
        velocities = self._entity.data.default_joint_vel_torch(env.device).index_select(0, env_ids)
        if env.torch_rng is None:
            raise NotImplementedError("FR3 tensor reset requires env.torch_rng")
        noise = env.torch_rng.generator
        position_noise = (
            torch.rand(positions.shape, generator=noise, device=positions.device)
            * (self._ranges[0][1] - self._ranges[0][0])
            + self._ranges[0][0]
        )
        velocity_noise = (
            torch.rand(velocities.shape, generator=noise, device=velocities.device)
            * (self._ranges[1][1] - self._ranges[1][0])
            + self._ranges[1][0]
        )
        self._entity.write_joint_state_tensor_to_sim(
            positions + position_noise,
            velocities + velocity_noise,
            env_ids=env_ids.to(dtype=torch.int64),
        )
