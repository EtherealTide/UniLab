from __future__ import annotations

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner
from unilab.rl.algos.sac.learner import SACLearner


def _sac() -> SACLearner:
    return SACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=5,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        use_layer_norm=False,
        max_grad_norm=0.0,
    )


def _flash_sac() -> FlashSACLearner:
    return FlashSACLearner(
        obs_dim=4,
        action_dim=2,
        critic_obs_dim=6,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        use_compile=False,
    )


@pytest.mark.parametrize("learner", [_sac(), _flash_sac()], ids=["sac", "flash_sac"])
def test_whole_cycle_adapter_contracts(learner: SACLearner | FlashSACLearner) -> None:
    optimizers = learner._whole_cycle_optimizers()
    modules = learner._whole_cycle_modules()
    target, source = learner._whole_cycle_target_modules()

    assert len(optimizers) == 3
    assert len(modules) == (3 if isinstance(learner, SACLearner) else 4)
    assert target is not source
    assert learner._whole_cycle_finite_gate().item() == 1.0
    learner._begin_update_cycle_metrics()
    assert learner._pending_actor_metric_values is None
    assert learner._pending_cycle_critic_metric_values is None
    assert learner._pending_cycle_metric_values is None


@pytest.mark.parametrize("learner", [_sac(), _flash_sac()], ids=["sac", "flash_sac"])
def test_materialize_capturable_state_restores_hyperparameters(
    learner: SACLearner | FlashSACLearner,
) -> None:
    learner._compile_full_update_cycle = True
    expected_weight_decay = [
        [group["weight_decay"] for group in optimizer.param_groups]
        for optimizer in learner._whole_cycle_optimizers()
    ]

    learner._materialize_capturable_optimizer_state()

    for optimizer, expected_groups in zip(
        learner._whole_cycle_optimizers(), expected_weight_decay, strict=True
    ):
        assert [group["lr"] for group in optimizer.param_groups] != [0.0]
        assert [group["weight_decay"] for group in optimizer.param_groups] == expected_groups
        assert optimizer.state
        assert all(
            torch.count_nonzero(value) == 0 if isinstance(value, torch.Tensor) else True
            for state in optimizer.state.values()
            for value in state.values()
        )
