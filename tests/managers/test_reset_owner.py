"""Contract tests for the Manager-owned fused reset-owner declaration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch

from unilab.managers import ResetOwner, ResetOwnerCfg, ResetOwnerManager
from unilab.managers._types import ManagerBasedRlEnv

from .conftest import FakeEnv


class RecordingResetOwner(ResetOwner):
    def __init__(self, cfg: ResetOwnerCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self.masks: list[torch.Tensor] = []

    def reset_transaction(self, reset_mask: torch.Tensor) -> dict[str, float]:
        assert isinstance(reset_mask, torch.Tensor)
        self.masks.append(reset_mask.clone())
        return {}


@dataclass(kw_only=True)
class RecordingResetOwnerCfg(ResetOwnerCfg):
    def build(self, env: ManagerBasedRlEnv) -> ResetOwner:
        return RecordingResetOwner(self, cast("ManagerBasedRlEnv", env))


def test_reset_owner_defaults_declare_ownership_explicitly() -> None:
    cfg = ResetOwnerCfg(func=RecordingResetOwner)
    assert cfg.command_name == "motion"
    assert cfg.owns_command_reset
    assert cfg.owns_action_reset
    assert cfg.owns_observation_reset
    assert cfg.owns_metric_reset


def test_reset_owner_manager_constructs_sole_owner_and_resets_mask() -> None:
    env = FakeEnv()
    manager = ResetOwnerManager({"motion": RecordingResetOwnerCfg(func=RecordingResetOwner)}, env)
    reset_mask = torch.tensor([False, True, False, True])
    manager.reset_transaction(reset_mask)
    owner = manager.get_term("motion")

    assert isinstance(owner, RecordingResetOwner)
    assert len(owner.masks) == 1
    torch.testing.assert_close(owner.masks[0], reset_mask)
    assert manager.active_terms == ["motion"]
    assert manager.owner is owner


def test_reset_owner_manager_rejects_multiple_owners() -> None:
    with pytest.raises(ValueError, match="at most one reset owner"):
        ResetOwnerManager(
            {
                "motion": RecordingResetOwnerCfg(func=RecordingResetOwner),
                "second": RecordingResetOwnerCfg(func=RecordingResetOwner),
            },
            FakeEnv(),
        )


def test_reset_owner_manager_rejects_non_owner_builder() -> None:
    @dataclass(kw_only=True)
    class BadCfg(ResetOwnerCfg):
        def build(self, env: ManagerBasedRlEnv) -> object:
            return object()

    with pytest.raises(TypeError, match="expected ResetOwner"):
        ResetOwnerManager({"bad": BadCfg(func=lambda cfg, env: object())}, FakeEnv())


def test_base_reset_owner_fails_closed() -> None:
    owner = ResetOwner(ResetOwnerCfg(func=ResetOwner), FakeEnv())
    with pytest.raises(NotImplementedError):
        owner.reset_transaction(torch.ones(2, dtype=torch.bool))
    with pytest.raises(NotImplementedError):
        owner.reset_committed(torch.ones(2, dtype=torch.bool))
