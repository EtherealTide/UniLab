"""Selected-row (env_ids) reset tests for ObservationManager."""

from __future__ import annotations

import pytest
import torch

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg

from .conftest import FakeEnv


class IdsTerm:
    def __init__(self) -> None:
        self.reset_ids: list[torch.Tensor] = []

    def __call__(self, env: FakeEnv) -> torch.Tensor:
        return env.obs.clone()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.reset_ids.append(env_ids.clone())


def test_selected_reset_passes_env_ids_and_invalidates_cache() -> None:
    env = FakeEnv(seed=3)
    source_term = IdsTerm()
    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=source_term)})},
        env,
    )
    term = manager.get_term_cfg("policy", "state").func
    assert isinstance(term, IdsTerm)
    full_obs = manager.compute(update_history=True)
    assert manager._obs_buffer is not None

    env_ids = torch.tensor([1, 3], dtype=torch.int64)
    assert manager.reset(env_ids) == {}
    assert manager._obs_buffer is None
    assert len(term.reset_ids) == 1
    torch.testing.assert_close(term.reset_ids[0], env_ids)

    reset_obs = manager.compute(update_history=True)
    assert reset_obs["policy"].shape == full_obs["policy"].shape


def test_reset_path_compute_requires_update_history_and_stays_full_width() -> None:
    env = FakeEnv(seed=3)
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: env.obs)}
            )
        },
        env,
    )
    full = manager.compute(update_history=True)
    ids = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(ValueError, match="update_history=True"):
        manager.compute(update_history=False, env_ids=ids)

    manager.reset(ids)
    rows = manager.compute(update_history=True, env_ids=ids)
    # The reset path publishes full-width rows; the env merges them through
    # the reset mask. The observation cache stays invalidated until the next
    # per-step compute refreshes it.
    assert rows["policy"].shape == full["policy"].shape
    assert manager._obs_buffer is None
    manager.compute(update_history=True)
    assert manager._obs_buffer is not None


@pytest.mark.parametrize(
    ("field", "value", "expected_dim"),
    (("delay_max_lag", 1, (2,)), ("history_length", 2, (4,))),
)
def test_temporal_terms_construct_and_report_temporal_dims(
    field: str, value: int, expected_dim: tuple[int, ...]
) -> None:
    env = FakeEnv(seed=3)
    term = ObservationTermCfg(func=lambda env: env.obs, **{field: value})
    manager = ObservationManager({"policy": ObservationGroupCfg(terms={"state": term})}, env)

    assert manager.group_obs_dim["policy"] == expected_dim
    obs = manager.compute(update_history=True)["policy"]
    assert obs.shape == (env.num_envs, *expected_dim)
    assert manager.reset(torch.tensor([0, 2], dtype=torch.int64)) == {}


def test_invalid_temporal_config_fails_clearly() -> None:
    env = FakeEnv(seed=3)
    with pytest.raises(ValueError, match="negative history_length"):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={"bad": ObservationTermCfg(func=lambda env: env.obs, history_length=-1)}
                )
            },
            env,
        )
    with pytest.raises(ValueError, match=r"invalid delay range \[2, 1\]"):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={
                        "bad": ObservationTermCfg(
                            func=lambda env: env.obs, delay_min_lag=2, delay_max_lag=1
                        )
                    }
                )
            },
            env,
        )


def test_reset_rejects_bool_mask() -> None:
    env = FakeEnv(seed=3)
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: env.obs, history_length=2)}
            )
        },
        env,
    )
    manager.compute(update_history=True)

    # The v1.4 bool-mask form is removed: any bool tensor fails closed.
    with pytest.raises(TypeError, match="integer row-index tensor"):
        manager.reset(torch.ones(env.num_envs + 1, dtype=torch.bool))
    with pytest.raises(TypeError, match="integer row-index tensor"):
        manager.reset(torch.zeros(env.num_envs, dtype=torch.bool))


def test_partial_reset_temporal_group_backfills_and_preserves_rows() -> None:
    env = FakeEnv(seed=5)
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "state": ObservationTermCfg(func=lambda env: env.obs, history_length=3),
                "delayed": ObservationTermCfg(
                    func=lambda env: env.obs, delay_min_lag=1, delay_max_lag=1
                ),
            },
        ),
    }
    manager = ObservationManager(cfg, env)
    for _ in range(4):
        env.obs = env.obs + 1
        manager.compute(update_history=True)

    ids = torch.tensor([1], dtype=torch.int64)
    keep_ids = torch.tensor([0, 2, 3], dtype=torch.int64)
    history_buffer = manager._group_obs_term_history_buffer["policy"]["state"]
    delay_buffer = manager._group_obs_term_delay_buffer["policy"]["delayed"]
    history_before = history_buffer.buffer.clone()
    delay_before = delay_buffer.peek().clone()

    manager.reset(ids)
    out = manager.compute(update_history=True, env_ids=ids)["policy"]

    # The publication stays full-width: 3x2 history slots + 2 delayed columns.
    assert out.shape == (env.num_envs, 8)
    # Untouched rows keep their history and delayed values bit-identically;
    # the reset row is backfilled with its post-reset frame in every slot.
    history_after = history_buffer.buffer
    torch.testing.assert_close(history_after[keep_ids], history_before[keep_ids])
    reset_slots = history_after[ids[0]]
    torch.testing.assert_close(reset_slots, reset_slots[-1].expand_as(reset_slots))
    torch.testing.assert_close(reset_slots[-1], env.obs[ids[0]])
    torch.testing.assert_close(delay_buffer.peek()[keep_ids], delay_before[keep_ids])


def _stochastic_delay_cfg() -> dict[str, ObservationGroupCfg]:
    return {
        "policy": ObservationGroupCfg(
            terms={
                "delayed": ObservationTermCfg(
                    func=lambda env: env.obs,
                    delay_min_lag=0,
                    delay_max_lag=3,
                    delay_hold_prob=0.25,
                    delay_update_period=3,
                    delay_per_env_phase=True,
                ),
            },
        ),
    }


def _run_delay_rollout(reset_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    env = FakeEnv(seed=7)
    manager = ObservationManager(_stochastic_delay_cfg(), env)
    for _ in range(4):
        manager.compute(update_history=True)
    manager.reset(reset_ids)
    manager.compute(update_history=True, env_ids=reset_ids)
    for _ in range(3):
        out = manager.compute(update_history=True)["policy"]
    buffer = manager._group_obs_term_delay_buffer["policy"]["delayed"]
    return (
        out,
        buffer.current_lags,
        buffer._step_count,
        buffer._phase_offsets,
        env.rng.generator.get_state(),
    )


def test_partial_reset_leaves_delay_schedule_and_lag_rng_untouched() -> None:
    """A partial reset is invisible to non-reset rows' delay timelines.

    Compared against a run resetting an empty selection, the delay schedule,
    lag values, phase offsets, outputs, and the shared generator state of the
    non-reset rows stay bit-identical: the reset boundary backfills and peeks
    instead of appending.
    """
    ids = torch.tensor([1], dtype=torch.int64)
    empty = torch.empty(0, dtype=torch.int64)
    keep = torch.tensor([0, 2, 3], dtype=torch.int64)

    out_reset, lags_reset, steps_reset, phases_reset, rng_reset = _run_delay_rollout(ids)
    out_empty, lags_empty, steps_empty, phases_empty, rng_empty = _run_delay_rollout(empty)

    torch.testing.assert_close(out_reset[keep], out_empty[keep])
    torch.testing.assert_close(lags_reset[keep], lags_empty[keep])
    torch.testing.assert_close(steps_reset[keep], steps_empty[keep])
    torch.testing.assert_close(phases_reset[keep], phases_empty[keep])
    assert torch.equal(rng_reset, rng_empty)
    # The reset row itself restarted its schedule.
    assert steps_reset[ids[0]] != steps_empty[ids[0]]


def test_removed_reset_row_contract_fails_closed_at_construction() -> None:
    class RemovedRowTerm(IdsTerm):
        def compute_reset_rows(self, env: FakeEnv, env_ids: torch.Tensor) -> torch.Tensor:
            return env.obs[env_ids]

    env = FakeEnv(seed=3)
    with pytest.raises(NotImplementedError, match="removed observation reset-row contract"):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={"state": ObservationTermCfg(func=RemovedRowTerm())}
                )
            },
            env,
        )
