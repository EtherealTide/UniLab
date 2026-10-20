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


def test_compute_is_always_full_width() -> None:
    env = FakeEnv(seed=3)
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: env.obs)}
            )
        },
        env,
    )
    obs = manager.compute(update_history=True)
    assert obs["policy"].shape == (env.num_envs, env.obs.shape[1])
    with pytest.raises(TypeError):
        manager.compute(update_history=True, env_ids=torch.tensor([1], dtype=torch.int64))  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("field", "value"),
    (("delay_max_lag", 1), ("history_length", 2)),
)
def test_temporal_terms_fail_closed_at_construction(field: str, value: int) -> None:
    env = FakeEnv(seed=3)
    term = ObservationTermCfg(func=lambda _: env.obs, **{field: value})
    with pytest.raises(NotImplementedError, match="fixed-width mask reset boundary"):
        ObservationManager({"policy": ObservationGroupCfg(terms={"state": term})}, env)


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
