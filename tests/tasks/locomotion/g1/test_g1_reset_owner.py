"""G1 walk fused mask-reset owner contract."""

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
        torch_rng=object(),
        action_manager=SimpleNamespace(cleared=[]),
        metrics_manager=SimpleNamespace(cleared=[]),
        device=torch.device("cpu"),
        num_envs=4,
    )
    env.command_manager.get_term = lambda name: command
    return env


def test_reset_transaction_uses_owner_mask_boundary() -> None:
    env = _env()
    seen: dict[str, object] = {}

    def reset(reset_mask):
        seen["mask"] = reset_mask

    command = env.command_manager.get_term("twist")
    command.reset = reset
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)

    mask = torch.tensor([False, True, False, True])

    assert owner.reset_transaction(mask) == {}
    assert seen["mask"] is mask


def test_reset_committed_mask_requires_action_manager_boundary() -> None:
    env = _env()
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)
    mask = torch.tensor([False, True, False, True])

    with pytest.raises(AttributeError, match="clear_action_state"):
        owner.reset_committed(mask)


def test_reset_committed_mask_clears_action_and_metric_state() -> None:
    env = _env()
    mask = torch.tensor([False, True, False, True])
    cleared_actions: list[torch.Tensor] = []
    cleared_metrics: list[torch.Tensor] = []

    def clear_action_state(reset_mask):
        cleared_actions.append(reset_mask)

    def clear_episode_state(reset_mask):
        cleared_metrics.append(reset_mask)

    env.action_manager.clear_action_state = clear_action_state
    env.metrics_manager.clear_episode_state = clear_episode_state
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)

    assert owner.reset_committed(mask) == {}
    assert cleared_actions == [mask]
    assert cleared_metrics == [mask]


def test_owner_validates_command_and_rng_contract() -> None:
    cfg = G1WalkResetOwnerCfg(func=lambda **kwargs: None)
    env = _env()
    env.command_manager.get_term = lambda name: SimpleNamespace(spec=object)
    with pytest.raises(TypeError, match="UniformVelocityCommand"):
        G1WalkResetOwner(cfg, env)

    env = _env()
    env.torch_rng = None
    with pytest.raises(NotImplementedError, match="Torch RNG"):
        G1WalkResetOwner(cfg, env)
