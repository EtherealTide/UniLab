"""G1 walk fused selected-reset owner contract."""

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


def test_reset_transaction_skips_host_publication_and_validation() -> None:
    env = _env()
    seen: dict[str, object] = {}

    def reset_command_state(rows, *, publish_metrics, validate_commands):
        seen["rows"] = rows
        seen["publish_metrics"] = publish_metrics
        seen["validate_commands"] = validate_commands
        return {}, {}

    env.command_manager.reset_command_state = reset_command_state
    owner = G1WalkResetOwner(G1WalkResetOwnerCfg(func=lambda **kwargs: None), env)

    rows = torch.tensor([1, 3], dtype=torch.int64)
    assert owner.reset_transaction(rows) == {}
    torch.testing.assert_close(seen["rows"], rows)
    assert seen["publish_metrics"] is False
    assert seen["validate_commands"] is False


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
