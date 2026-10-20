"""G1 walk fused selected-row reset owner contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unilab.tasks.locomotion.g1.manager_terms import (
    G1VelocityCommand,
    G1WalkResetOwner,
    G1WalkResetOwnerCfg,
)


def _env() -> SimpleNamespace:
    command = G1VelocityCommand.__new__(G1VelocityCommand)
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(spec=object)),
        rng=object(),
        action_manager=SimpleNamespace(cleared=[]),
        metrics_manager=SimpleNamespace(cleared=[]),
        device=torch.device("cpu"),
        num_envs=4,
    )
    env.command_manager.get_term = lambda name: command
    return env


def test_owner_cfg_defaults_delegate_observation_reset() -> None:
    cfg = G1WalkResetOwnerCfg(func=lambda **kwargs: None)

    assert cfg.command_name == "twist"
    assert cfg.owns_observation_reset is False


def test_reset_transaction_uses_owner_env_ids_boundary() -> None:
    env = _env()
    seen: dict[str, object] = {}

    def reset(env_ids):
        seen["env_ids"] = env_ids

    command = env.command_manager.get_term("twist")
    command.reset = reset
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)

    env_ids = torch.tensor([1, 3], dtype=torch.int64)

    assert owner.reset_transaction(env_ids) == {}
    assert seen["env_ids"] is env_ids


def test_reset_committed_requires_action_manager_boundary() -> None:
    env = _env()
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)
    env_ids = torch.tensor([1, 3], dtype=torch.int64)

    with pytest.raises(AttributeError, match="clear_action_state"):
        owner.reset_committed(env_ids)


def test_reset_committed_clears_action_and_metric_state() -> None:
    env = _env()
    env_ids = torch.tensor([1, 3], dtype=torch.int64)
    cleared_actions: list[torch.Tensor] = []
    cleared_metrics: list[torch.Tensor] = []

    def clear_action_state(env_ids):
        cleared_actions.append(env_ids)

    def clear_episode_state(env_ids):
        cleared_metrics.append(env_ids)

    env.action_manager.clear_action_state = clear_action_state
    env.metrics_manager.clear_episode_state = clear_episode_state
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)

    assert owner.reset_committed(env_ids) == {}
    assert cleared_actions == [env_ids]
    assert cleared_metrics == [env_ids]


def test_owner_validates_command_and_rng_contract() -> None:
    cfg = G1WalkResetOwnerCfg(func=lambda **kwargs: None)
    env = _env()
    env.command_manager.get_term = lambda name: SimpleNamespace(spec=object)
    with pytest.raises(TypeError, match="UniformVelocityCommand"):
        G1WalkResetOwner(cfg, env)

    env = _env()
    env.rng = None
    with pytest.raises(NotImplementedError, match="Manager RNG"):
        G1WalkResetOwner(cfg, env)
