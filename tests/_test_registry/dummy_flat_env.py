"""DummyFlatTest env registration — importable both from pytest conftest (parent
process) and from the spawn collector subprocesses via
``UNILAB_EXTRA_REGISTRY_PACKAGES`` + ``ensure_registries``.

The env is a concrete ``TorchEnv`` so off-policy/APPO collector loops drive the
same tensor lifecycle as production tasks. The dynamics are intentionally
trivial: deterministic finite observations, zero reward, never-done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import torch
from unisim.backend.base import (
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
)

from unilab.base import registry
from unilab.base.base import EnvCfg, EnvPlayCapabilities
from unilab.base.torch_env import TorchEnv

_DUMMY_OBS_DIM = 8
_DUMMY_ACT_DIM = 3
DUMMY_ENV_NAME = "DummyFlatTest"


@dataclass
class _DummyCfg(EnvCfg):
    pass


def _dummy_backend() -> MagicMock:
    backend = MagicMock(spec=SimBackend)
    backend.backend_type = "mujoco"
    backend.tensor_execution.return_value = TensorExecution.HOST_BRIDGE
    backend.get_tensor_capabilities.return_value = TensorLifecycleCapabilities(
        execution=TensorExecution.HOST_BRIDGE,
        state_fields=frozenset({"qpos", "qvel"}),
        stepping=True,
        selected_reset=True,
        packed_host_bridge=True,
        data_plane=TensorDataPlane.HOST_BRIDGE,
        stream_event_ownership="dummy-flat-test",
        torch_devices=("cpu",),
    )
    backend.step_tensor.return_value = None
    return backend


class _DummyEnv(TorchEnv):
    """Minimal tensor env: deterministic finite obs, zero reward, never done."""

    def __init__(self, cfg: _DummyCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        super().__init__(cfg, _dummy_backend(), num_envs, device="cpu")
        self._obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_DUMMY_OBS_DIM,), dtype=np.float32
        )
        self._act_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_DUMMY_ACT_DIM,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._obs_space

    @property
    def action_space(self) -> gym.Space:
        return self._act_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": _DUMMY_OBS_DIM}

    @property
    def play_capabilities(self) -> EnvPlayCapabilities:
        return EnvPlayCapabilities(supports_physics_state_playback=False)

    def _obs(self) -> dict[str, torch.Tensor]:
        rows = self.num_envs
        return {
            "obs": torch.linspace(
                -1.0, 1.0, rows * _DUMMY_OBS_DIM, dtype=torch.float32, device=self.device
            ).reshape(rows, _DUMMY_OBS_DIM)
        }

    def reset(self, env_indices: torch.Tensor | None = None):
        rows = self._normalize_reset_indices(env_indices)
        return self._obs_for_rows(rows.numel()), {}

    def _obs_for_rows(self, count: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.linspace(
                -1.0, 1.0, count * _DUMMY_OBS_DIM, dtype=torch.float32, device=self.device
            ).reshape(count, _DUMMY_OBS_DIM)
        }

    def apply_action(self, actions: torch.Tensor, state: Any) -> torch.Tensor:
        return actions

    def update_state(self, state: Any):
        return state.replace(
            obs=self._obs(),
            reward=torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            terminated=torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            truncated=torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
            info={"steps": state.info.get("steps", torch.zeros(self.num_envs, dtype=torch.int64))},
        )


def register() -> None:
    if not registry.contains(DUMMY_ENV_NAME):
        registry.register_env_config(DUMMY_ENV_NAME, _DummyCfg)
        registry.register_env(DUMMY_ENV_NAME, _DummyEnv, "mujoco")


register()
