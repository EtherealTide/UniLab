"""Fake env implementations for unilab.rl runner/collector unit tests.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15) tests/conftest.py.

These classes live in a plain importable module (not a conftest) so that
spawn-based collector subprocesses can pickle factories by reference:
``tests.rl._fakes.FakeEnvFactory`` resolves in any interpreter with the repo
root on ``sys.path``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


class _FakeSpace:
    """Gym-style space stub; the env contract only reads ``.shape``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


@dataclass
class _FakeEnvState:
    """Minimal ``EnvStateProtocol`` implementation (random obs, zero reward)."""

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    final_observation: dict[str, np.ndarray] | None = None

    def replace(self, **updates: Any) -> "_FakeEnvState":
        return dataclasses.replace(self, **updates)


class _FakePlayCapabilities:
    supports_physics_state_playback = False


class FakeVecEnv:
    """Minimal ``EnvProtocol`` implementation for runner/collector unit tests.

    Pass ``algo_capabilities`` to make the env satisfy
    ``SupportsAlgoCapabilitiesProtocol``; by default the property is absent
    and ``get_algo_capabilities`` falls back to the all-``None`` default.
    """

    def __init__(
        self,
        num_envs: int,
        obs_groups_spec: dict[str, int],
        action_dim: int,
        algo_capabilities: Any = None,
    ) -> None:
        self._num_envs = num_envs
        self._obs_groups_spec = dict(obs_groups_spec)
        self._action_dim = action_dim
        self._state: _FakeEnvState | None = None
        self.closed = False
        self.cfg = type("Cfg", (), {"max_episode_seconds": 1.0, "ctrl_dt": 0.02})()
        if algo_capabilities is not None:
            self.algo_capabilities = algo_capabilities

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return dict(self._obs_groups_spec)

    @property
    def observation_space(self) -> _FakeSpace:
        return _FakeSpace((sum(self._obs_groups_spec.values()),))

    @property
    def action_space(self) -> _FakeSpace:
        return _FakeSpace((self._action_dim,))

    @property
    def state(self) -> _FakeEnvState | None:
        return self._state

    @property
    def play_capabilities(self) -> _FakePlayCapabilities:
        return _FakePlayCapabilities()

    def init_state(self) -> _FakeEnvState:
        self._state = _FakeEnvState(
            obs={
                k: np.zeros((self._num_envs, d), dtype=np.float32)
                for k, d in self._obs_groups_spec.items()
            },
            reward=np.zeros((self._num_envs,), dtype=np.float32),
            terminated=np.zeros((self._num_envs,), dtype=bool),
            truncated=np.zeros((self._num_envs,), dtype=bool),
            info={"steps": np.zeros((self._num_envs,), dtype=np.uint32)},
        )
        return self._state

    def step(self, actions: np.ndarray) -> _FakeEnvState:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        return self._state

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        if self._state is None:
            self.init_state()
        n = len(env_indices)
        return (
            {k: np.zeros((n, d), dtype=np.float32) for k, d in self._obs_groups_spec.items()},
            {},
        )

    def set_nan_guard(self, guard: Any) -> None:
        del guard

    def close(self) -> None:
        self.closed = True


class FakeEnvFactory:
    """Picklable ``EnvFactory`` producing :class:`FakeVecEnv` probes.

    Top-level class so it survives spawn-process pickling like a real
    registry-backed factory would.
    """

    def __init__(
        self,
        obs_groups_spec: dict[str, int] | None = None,
        action_dim: int = 2,
        algo_capabilities: Any = None,
    ) -> None:
        self.obs_groups_spec = dict(obs_groups_spec or {"obs": 4, "critic": 7})
        self.action_dim = action_dim
        self.algo_capabilities = algo_capabilities
        self.calls: list[tuple[int, Any]] = []

    def __call__(self, num_envs: int, env_cfg_override: Any = None) -> FakeVecEnv:
        self.calls.append((num_envs, env_cfg_override))
        return FakeVecEnv(num_envs, self.obs_groups_spec, self.action_dim, self.algo_capabilities)


@dataclass
class StrictTensorEnvState:
    obs: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any] = field(default_factory=dict)
    final_observation: dict[str, torch.Tensor] | None = None


class StrictTensorEnv:
    """Tensor-native env that rejects NumPy actions/reset indices."""

    def __init__(self, num_envs: int = 1, obs_dim: int = 3, action_dim: int = 2) -> None:
        self._num_envs = num_envs
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._device = torch.device("cpu")
        self._step_index = 0
        self._state: StrictTensorEnvState | None = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._obs_dim, "critic": self._obs_dim + 1}

    @property
    def observation_space(self) -> _FakeSpace:
        return _FakeSpace((self._obs_dim + self._obs_dim + 1,))

    @property
    def action_space(self) -> _FakeSpace:
        return _FakeSpace((self._action_dim,))

    @property
    def state(self) -> StrictTensorEnvState | None:
        return self._state

    @property
    def cfg(self) -> Any:
        return type("Cfg", (), {"max_episode_seconds": 1.0, "ctrl_dt": 0.02})()

    @property
    def play_capabilities(self) -> _FakePlayCapabilities:
        return _FakePlayCapabilities()

    def _obs(self) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.full(
                (self._num_envs, self._obs_dim), float(self._step_index), dtype=torch.float32
            ),
            "critic": torch.full(
                (self._num_envs, self._obs_dim + 1), float(self._step_index), dtype=torch.float32
            ),
        }

    def _new_state(self, terminal: bool) -> StrictTensorEnvState:
        return StrictTensorEnvState(
            obs=self._obs(),
            reward=torch.full((self._num_envs,), float(self._step_index), dtype=torch.float32),
            terminated=torch.full((self._num_envs,), terminal, dtype=torch.bool),
            truncated=torch.full((self._num_envs,), terminal, dtype=torch.bool),
            info={},
            final_observation=self._obs() if terminal else None,
        )

    def init_state(self) -> StrictTensorEnvState:
        self._step_index = 0
        self._state = self._new_state(False)
        return self._state

    def step(self, actions: torch.Tensor) -> StrictTensorEnvState:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"action must be torch.Tensor, got {type(actions).__name__}")
        if actions.dtype != torch.float32 or not actions.is_contiguous():
            raise TypeError("action must be contiguous float32")
        if actions.device != self._device or actions.shape != (
            self._num_envs,
            self._action_dim,
        ):
            raise ValueError("action device/shape mismatch")
        self._step_index += 1
        self._state = self._new_state(self._step_index >= 2)
        return self._state

    def reset(self, env_indices: torch.Tensor | None) -> tuple[dict[str, torch.Tensor], dict]:
        if env_indices is not None:
            if not isinstance(env_indices, torch.Tensor):
                raise TypeError("reset indices must be torch.Tensor")
            if (
                env_indices.dtype not in (torch.int32, torch.int64)
                or env_indices.device != self._device
            ):
                raise TypeError("reset indices must be integer tensors on env.device")
        return self._obs(), {}

    def set_nan_guard(self, guard: Any) -> None:
        del guard

    def close(self) -> None:
        pass


class StrictTensorEnvFactory:
    """Picklable factory for :class:`StrictTensorEnv`."""

    def __init__(self, obs_dim: int = 3, action_dim: int = 2) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def __call__(self, num_envs: int, env_cfg_override: Any = None) -> StrictTensorEnv:
        del env_cfg_override
        return StrictTensorEnv(num_envs, self.obs_dim, self.action_dim)
