# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), manager tests.
# Modified by UniLab for NumPy and fail-closed term validation; Apache-2.0.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from unilab.managers import (
    ActionManager,
    ActionTerm,
    ActionTermCfg,
    CurriculumManager,
    CurriculumTermCfg,
    NullCurriculumManager,
    RewardManager,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationManager,
    TerminationTermCfg,
)

from .conftest import FakeEnv


class DummyAction(ActionTerm):
    def __init__(self, cfg: DummyActionCfg, env: FakeEnv):
        super().__init__(cfg, env)
        self._raw = np.zeros((env.num_envs, cfg.dim), dtype=np.float32)
        self.applied = 0
        self.reset_ids: np.ndarray | slice | None = None
        self.input_types: list[type] = []

    @property
    def action_dim(self) -> int:
        return self._raw.shape[1]

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw

    def process_actions(self, actions: np.ndarray) -> None:
        self.input_types.append(type(actions))
        self._raw[:] = actions

    def apply_actions(self) -> None:
        self.applied += 1

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self.reset_ids = env_ids
        self._raw[env_ids] = 0.0


@dataclass(kw_only=True)
class DummyActionCfg(ActionTermCfg):
    dim: int

    def build(self, env: FakeEnv) -> DummyAction:
        return DummyAction(self, env)


class FeedbackDummyAction(DummyAction):
    requires_substep_state_feedback = True


@dataclass(kw_only=True)
class FeedbackDummyActionCfg(DummyActionCfg):
    def build(self, env: FakeEnv) -> FeedbackDummyAction:
        return FeedbackDummyAction(self, env)


def test_action_split_history_apply_and_partial_reset(fake_env: FakeEnv) -> None:
    manager = ActionManager(
        {
            "legs": DummyActionCfg(entity_name="robot", dim=2),
            "disabled": None,
            "arm": DummyActionCfg(entity_name="robot", dim=1),
        },
        fake_env,
    )
    assert manager.active_terms == ["legs", "arm"]
    assert not manager.requires_substep_state_feedback
    first = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    second = first + 20
    manager.process_action(first)
    manager.process_action(second)
    assert isinstance(manager.action, torch.Tensor)
    assert manager.action.dtype == torch.float32
    assert all(
        term.input_types and term.input_types[0] is np.ndarray
        for term in (manager.get_term("legs"), manager.get_term("arm"))
    )
    torch.testing.assert_close(manager.prev_action, first)
    torch.testing.assert_close(manager.action, second)
    np.testing.assert_array_equal(manager.get_term("legs").raw_action, second[:, :2])
    manager.apply_action()
    assert manager.get_term("legs").applied == 1
    manager.reset(np.array([1, 3]))
    torch.testing.assert_close(manager.action[[1, 3]], torch.zeros(2, 3))
    torch.testing.assert_close(manager.action[[0, 2]], second[[0, 2]])


def test_action_manager_aggregates_substep_state_feedback(fake_env: FakeEnv) -> None:
    manager = ActionManager(
        {
            "invariant": DummyActionCfg(entity_name="robot", dim=1),
            "feedback": FeedbackDummyActionCfg(entity_name="robot", dim=1),
        },
        fake_env,
    )

    assert manager.requires_substep_state_feedback


def test_action_feedback_declaration_must_be_bool(
    fake_env: FakeEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DummyAction, "requires_substep_state_feedback", "yes")

    with pytest.raises(TypeError, match="requires_substep_state_feedback must be bool"):
        ActionManager({"invalid": DummyActionCfg(entity_name="robot", dim=1)}, fake_env)


@pytest.mark.parametrize(
    "action,match",
    [
        (torch.zeros((4, 2), dtype=torch.float32), "Invalid action shape"),
        (torch.full((4, 3), torch.nan, dtype=torch.float32), "NaN or Inf"),
        (np.zeros((4, 3), dtype=np.float32), "torch.Tensor"),
        (torch.zeros((4, 3), dtype=torch.float64), "contiguous float32"),
        (torch.zeros((4, 6), dtype=torch.float32)[:, ::2], "contiguous float32"),
    ],
)
def test_action_rejects_invalid_input(fake_env: FakeEnv, action: np.ndarray, match: str) -> None:
    manager = ActionManager({"a": DummyActionCfg(entity_name="robot", dim=3)}, fake_env)
    with pytest.raises((TypeError, ValueError), match=match):
        manager.process_action(action)


class FailingAction(DummyAction):
    def process_actions(self, actions: np.ndarray) -> None:
        del actions
        raise ValueError("invalid processed target")

    def apply_actions(self) -> None:
        raise NotImplementedError("backend control write unavailable")


@dataclass(kw_only=True)
class FailingActionCfg(DummyActionCfg):
    def build(self, env: FakeEnv) -> FailingAction:
        return FailingAction(self, env)


def test_action_term_errors_include_manager_and_term_context(fake_env: FakeEnv) -> None:
    manager = ActionManager(
        {"broken": FailingActionCfg(entity_name="robot", dim=1)},
        fake_env,
    )
    with pytest.raises(ValueError, match="ActionManager term 'broken'.*invalid processed"):
        manager.process_action(torch.zeros((fake_env.num_envs, 1), dtype=torch.float32))
    with pytest.raises(NotImplementedError, match="ActionManager term 'broken'.*control write"):
        manager.apply_action()


class StatefulReward:
    def __init__(self, cfg: RewardTermCfg, env: FakeEnv):
        self.reset_ids = None

    def __call__(self, env: FakeEnv) -> np.ndarray:
        return env.value.copy()

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        self.reset_ids = env_ids


def test_reward_dt_scaling_reset_and_config_immutability(fake_env: FakeEnv) -> None:
    cfg = {"stateful": RewardTermCfg(func=StatefulReward, weight=2.0)}
    manager = RewardManager(cfg, fake_env)
    reward = manager.compute(dt=0.25)
    assert isinstance(reward, torch.Tensor)
    assert reward.device.type == "cpu"
    torch.testing.assert_close(reward, torch.tensor(fake_env.value * 0.5))
    assert manager.get_active_iterable_terms(2) == [("stateful", [4.0])]
    reset_ids = np.array([1, 2])
    extras = manager.reset(reset_ids)
    assert extras == {}
    assert not any(key.startswith("Episode_Reward/") for key in extras)
    assert manager.get_term_cfg("stateful").func.reset_ids is reset_ids
    assert cfg["stateful"].func is StatefulReward
    assert isinstance(manager.get_term_cfg("stateful").func, StatefulReward)


def test_reward_step_extras_report_per_term_weighted_rates(fake_env: FakeEnv) -> None:
    def ones(env: FakeEnv) -> np.ndarray:
        return np.ones(env.num_envs, dtype=np.float32)

    cfg = {
        "pos": RewardTermCfg(func=ones, weight=2.0),
        "neg": RewardTermCfg(func=lambda env: env.value, weight=-0.5),
        "zero": RewardTermCfg(func=ones, weight=0.0),
    }
    manager = RewardManager(cfg, fake_env)
    manager.compute(dt=0.25)

    extras = manager.step_reward_extras()
    assert set(extras) == {"reward/pos", "reward/neg", "reward/zero"}
    # Weighted reward rate (raw_value * weight), not scaled by dt.
    assert extras["reward/pos"] == pytest.approx(2.0)
    assert extras["reward/neg"] == pytest.approx(float(np.mean(fake_env.value)) * -0.5)
    assert extras["reward/zero"] == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reward_nonfinite_is_an_error(fake_env: FakeEnv, bad: float) -> None:
    def reward(env: FakeEnv) -> np.ndarray:
        value = np.ones(env.num_envs, dtype=np.float32)
        value[2] = bad
        return value

    manager = RewardManager({"bad_reward": RewardTermCfg(func=reward, weight=1.0)}, fake_env)
    with pytest.raises(ValueError, match="RewardManager term 'bad_reward'"):
        manager.compute(0.01)


def test_tensor_manager_terms_require_declared_dtype(fake_env: FakeEnv) -> None:
    reward = RewardManager(
        {
            "bad": RewardTermCfg(
                func=lambda env: torch.ones(env.num_envs, dtype=torch.float64), weight=1.0
            )
        },
        fake_env,
    )
    with pytest.raises(TypeError, match="expected float32"):
        reward.compute(0.1)

    termination = TerminationManager(
        {"bad": TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.uint8))},
        fake_env,
    )
    with pytest.raises(TypeError, match="expected bool"):
        termination.compute()


def test_reward_and_termination_shape_validation(fake_env: FakeEnv) -> None:
    reward = RewardManager(
        {"bad": RewardTermCfg(func=lambda env: np.zeros((env.num_envs, 1)), weight=1.0)},
        fake_env,
    )
    with pytest.raises(ValueError, match=r"expected \(4,\)"):
        reward.compute(0.1)

    termination = TerminationManager(
        {"bad": TerminationTermCfg(func=lambda env: np.zeros(env.num_envs))}, fake_env
    )
    with pytest.raises(TypeError, match="expected bool"):
        termination.compute()


def test_termination_splits_timeouts_and_failures(fake_env: FakeEnv) -> None:
    timeout = np.array([True, False, False, True])
    failure = np.array([False, True, False, True])
    manager = TerminationManager(
        {
            "timeout": TerminationTermCfg(func=lambda env: timeout.copy(), time_out=True),
            "failure": TerminationTermCfg(func=lambda env: failure.copy()),
        },
        fake_env,
    )
    dones = manager.compute()
    assert isinstance(dones, torch.Tensor)
    assert dones.dtype == torch.bool
    torch.testing.assert_close(dones, torch.from_numpy(timeout | failure))
    torch.testing.assert_close(manager.time_outs, torch.from_numpy(timeout))
    torch.testing.assert_close(manager.terminated, torch.from_numpy(failure))
    assert manager.reset(np.array([0, 1])) == {
        "Episode_Termination/timeout": 1,
        "Episode_Termination/failure": 1,
    }


def test_curriculum_and_null_semantics(fake_env: FakeEnv) -> None:
    def update(env: FakeEnv, env_ids: np.ndarray | slice) -> dict[str, float]:
        return {"difficulty": 3.0}

    manager = CurriculumManager({"terrain": CurriculumTermCfg(func=update)}, fake_env)
    manager.compute(np.array([1, 2]))
    assert manager.reset()["Curriculum/terrain/difficulty"] == 3.0
    null = NullCurriculumManager()
    assert null.active_terms == []
    assert null.reset() == {}

    bad = CurriculumManager({"bad": CurriculumTermCfg(func=lambda env, env_ids: np.nan)}, fake_env)
    with pytest.raises(ValueError, match="CurriculumManager term 'bad'"):
        bad.compute()


def test_scene_entity_selector_resolution(fake_env: FakeEnv) -> None:
    cfg = SceneEntityCfg(name="robot", joint_names=("ankle", "hip"), preserve_order=True)
    cfg.resolve(fake_env.scene)
    assert cfg.joint_names == ["ankle", "hip"]
    assert cfg.joint_ids == [2, 0]

    all_joints = SceneEntityCfg(name="robot", joint_names=".*")
    all_joints.resolve(fake_env.scene)
    assert all_joints.joint_ids == slice(None)

    inconsistent = SceneEntityCfg(name="robot", joint_names="hip", joint_ids=[1])
    with pytest.raises(ValueError, match="Inconsistent joint"):
        inconsistent.resolve(fake_env.scene)
