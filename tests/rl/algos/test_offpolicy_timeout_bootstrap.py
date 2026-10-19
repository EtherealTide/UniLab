"""Collector -> replay -> learner regression for timeout bootstrap eligibility.

Driving issue: https://github.com/unilabsim/unilab_rl/issues/29

When one transition returns ``terminated=True`` and ``truncated=True``
together, the failure must override the timeout: the row is a true terminal
and must not bootstrap. These tests drive the real collector write boundary
(``_run_collector``) into a real ``ReplayBuffer`` and feed the stored rows to
the real SAC and FlashSAC critic updates, asserting the full truth table:

| terminated | truncated | done | bootstrap |
|------------|-----------|------|-----------|
| False      | False     | 0    | 1         |
| True       | False     | 1    | 0         |
| False      | True      | 1    | 1 (terminal obs) |
| True       | True      | 1    | 0         |
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

import unilab.rl.runtime.offpolicy.worker as worker_module
from unilab.rl.algos.flash_sac import learner as flash_sac_module
from unilab.rl.algos.flash_sac.learner import FlashSACLearner, RewardNormalizer
from unilab.rl.algos.sac.learner import SACLearner
from unilab.rl.runtime.ipc.inference_ring import SharedInferenceRing
from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer

_NUM_ENVS = 4
_OBS_DIM = 2
_CRITIC_DIM = 3
_ACTION_DIM = 2

_REWARDS = torch.tensor([1.0, 2.0, 3.0, 4.0])
_TERMINATED = torch.tensor([False, True, False, True])
_TRUNCATED = torch.tensor([False, False, True, True])
_EXPECTED_DONES = torch.tensor([0.0, 1.0, 1.0, 1.0])
_EXPECTED_TIMEOUT_BOOTSTRAP = torch.tensor([0.0, 0.0, 1.0, 0.0])
_EXPECTED_BOOTSTRAP = torch.tensor([1.0, 0.0, 1.0, 0.0])


def _row_values(base: float, width: int) -> torch.Tensor:
    return torch.arange(_NUM_ENVS, dtype=torch.float32).unsqueeze(1).expand(_NUM_ENVS, width) + base


# Post-reset (autoreset) observation rows: must never be bootstrapped from.
_NEXT_OBS = _row_values(100.0, _OBS_DIM)
_NEXT_CRITIC = _row_values(200.0, _CRITIC_DIM)
# Real terminal observation rows: the only valid bootstrap source.
_TERMINAL_OBS = _row_values(10.0, _OBS_DIM)
_TERMINAL_CRITIC = _row_values(20.0, _CRITIC_DIM)


class _InspectableReplayBuffer(ReplayBuffer):
    """ReplayBuffer that stays readable after the collector teardown."""

    def release_ipc(self) -> None:
        # The production collector releases its ingress IPC views on exit; the
        # test inspects the written rows afterwards from the "learner" side.
        pass


def _collect_one_step(monkeypatch: pytest.MonkeyPatch) -> tuple[ReplayBuffer, torch.Tensor]:
    """Run the real collector for one 4-row step covering every combination."""
    stop_event = threading.Event()

    initial_state = SimpleNamespace(
        obs={
            "obs": torch.zeros((_NUM_ENVS, _OBS_DIM)),
            "critic": torch.zeros((_NUM_ENVS, _CRITIC_DIM)),
        },
        reward=torch.zeros(_NUM_ENVS),
        terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        final_observation=None,
        info={"timing": {}},
    )

    class _Env:
        state = initial_state

        def step(self, actions):
            torch.testing.assert_close(actions, torch.zeros((_NUM_ENVS, _ACTION_DIM)))
            stop_event.set()
            self.state = SimpleNamespace(
                obs={"obs": _NEXT_OBS.clone(), "critic": _NEXT_CRITIC.clone()},
                reward=_REWARDS.clone(),
                terminated=_TERMINATED.clone(),
                truncated=_TRUNCATED.clone(),
                final_observation={
                    "obs": _TERMINAL_OBS.clone(),
                    "critic": _TERMINAL_CRITIC.clone(),
                },
                info={"timing": {}},
            )
            return self.state

        def close(self) -> None:
            pass

    replay = _InspectableReplayBuffer(
        capacity=16,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        device="cpu",
        ingress_slot_rows=_NUM_ENVS,
        critic_dim=_CRITIC_DIM,
    )
    ring = SharedInferenceRing(_NUM_ENVS, _OBS_DIM, _ACTION_DIM, capacity=2)

    def serve_learner_action(coordination_queue, tick_id, stop, **kwargs):
        del coordination_queue, stop, kwargs
        ring.copy_observation_to(
            tick_id=tick_id,
            observations=torch.empty((_NUM_ENVS, _OBS_DIM)),
            dones=torch.empty(_NUM_ENVS),
            epoch=0,
        )
        ring.publish_action(
            tick_id=tick_id,
            policy_version=0,
            actions=torch.zeros((_NUM_ENVS, _ACTION_DIM)),
            epoch=0,
        )
        return True

    monkeypatch.setattr(worker_module, "apply_torch_thread_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "apply_worker_seed", lambda *a, **kw: None)
    monkeypatch.setattr(worker_module, "_wait_for_inference_tick", serve_learner_action)
    worker_module._run_collector(
        stop_event=stop_event,
        env_factory=lambda num_envs, env_cfg_override=None: _Env(),
        num_envs=_NUM_ENVS,
        replay_buffer=replay,
        inference_slot=ring,
        inference_request_queue=queue.Queue(),
        inference_response_queue=queue.Queue(),
        algo_type="sac",
        actor_adapter_modules=None,
        metrics_queue=None,
        inference_transport="host",
        sim_backend="mujoco",
        backend_device=None,
        env_cfg_override=None,
        inference_epoch=0,
        collector_metrics_interval=1,
        seed=1,
        trace_enabled=False,
        trace_thread_time=False,
    )

    ingress = replay.take_published_ingress()
    assert ingress is not None
    slot, start, count, packed = ingress
    assert count == _NUM_ENVS
    replay.commit_ingress(slot=slot, start=start, count=count)
    return replay, packed


def _replay_batch(replay: ReplayBuffer, packed: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "obs": replay.field_view(packed, "obs"),
        "next_obs": replay.field_view(packed, "next_obs"),
        "actions": replay.field_view(packed, "actions"),
        "rewards": replay.field_view(packed, "rewards"),
        "dones": replay.field_view(packed, "dones"),
        "truncated": replay.field_view(packed, "truncated"),
        "critic": replay.field_view(packed, "critic"),
        "next_critic": replay.field_view(packed, "next_critic"),
    }


@pytest.fixture()
def collected_batch(monkeypatch: pytest.MonkeyPatch) -> dict[str, torch.Tensor]:
    replay, packed = _collect_one_step(monkeypatch)
    return _replay_batch(replay, packed)


def test_collector_encodes_timeout_bootstrap_eligibility_at_replay_boundary(
    collected_batch: dict[str, torch.Tensor],
) -> None:
    # Episode lifecycle is unchanged: done = terminated | truncated.
    torch.testing.assert_close(collected_batch["dones"], _EXPECTED_DONES)
    # The replay `truncated` column carries truncated & ~terminated, not the
    # raw truncation flags: the overlap row (index 3) must not be eligible.
    torch.testing.assert_close(collected_batch["truncated"], _EXPECTED_TIMEOUT_BOOTSTRAP)
    # Terminal transition rewards are preserved exactly; no bootstrap value is
    # folded into the stored reward.
    torch.testing.assert_close(collected_batch["rewards"], _REWARDS)


def test_replay_done_rows_carry_terminal_observations_not_reset_observations(
    collected_batch: dict[str, torch.Tensor],
) -> None:
    expected_next_obs = torch.where(_EXPECTED_DONES.unsqueeze(1) > 0.5, _TERMINAL_OBS, _NEXT_OBS)
    expected_next_critic = torch.where(
        _EXPECTED_DONES.unsqueeze(1) > 0.5, _TERMINAL_CRITIC, _NEXT_CRITIC
    )
    torch.testing.assert_close(collected_batch["next_obs"], expected_next_obs)
    torch.testing.assert_close(collected_batch["next_critic"], expected_next_critic)
    # The timeout-only row bootstraps from its terminal observation, never
    # from the post-reset observation.
    torch.testing.assert_close(collected_batch["next_obs"][2], _TERMINAL_OBS[2])
    torch.testing.assert_close(collected_batch["next_critic"][2], _TERMINAL_CRITIC[2])


class _CaptureSacTargetCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
        self.bootstrap: torch.Tensor | None = None
        self.critic_next_obs: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None

    def projection(self, obs, actions, rewards, bootstrap, discount):
        del actions, discount
        self.bootstrap = bootstrap.detach().clone()
        self.critic_next_obs = obs.detach().clone()
        self.rewards = rewards.detach().clone()
        batch_size = bootstrap.shape[0]
        probs = torch.zeros(2, batch_size, 3, dtype=bootstrap.dtype)
        probs[:, :, 0] = 1.0
        return probs

    def get_value(self, probs):
        return probs[:, :, 0]


def test_sac_critic_failure_overrides_timeout_bootstrap(
    collected_batch: dict[str, torch.Tensor],
) -> None:
    learner = SACLearner(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_obs_dim=_CRITIC_DIM,
        device="cpu",
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        num_atoms=3,
        num_q_networks=2,
        use_layer_norm=False,
        use_autotune=False,
    )
    target_critic = _CaptureSacTargetCritic()
    learner.qnet_target = target_critic

    learner.update_critic(collected_batch)

    assert target_critic.bootstrap is not None
    torch.testing.assert_close(target_critic.bootstrap, _EXPECTED_BOOTSTRAP)
    # Failure rows (terminated, including the overlap row) are pure terminal
    # targets: the stored reward reaches the projection unmodified.
    assert target_critic.rewards is not None
    torch.testing.assert_close(target_critic.rewards[1], _REWARDS[1])
    torch.testing.assert_close(target_critic.rewards[3], _REWARDS[3])
    # The timeout-only row bootstraps from the real terminal critic
    # observation delivered through replay.
    assert target_critic.critic_next_obs is not None
    torch.testing.assert_close(target_critic.critic_next_obs[2], _TERMINAL_CRITIC[2])


def test_flashsac_critic_failure_overrides_timeout_bootstrap(
    collected_batch: dict[str, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}
    original = flash_sac_module.compute_categorical_td_target

    def spy(**kwargs):
        captured.update(
            reward=kwargs["reward"].detach().clone(),
            dones=kwargs["dones"].detach().clone(),
            truncated=kwargs["truncated"].detach().clone(),
        )
        return original(**kwargs)

    monkeypatch.setattr(flash_sac_module, "compute_categorical_td_target", spy)
    learner = FlashSACLearner(
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_obs_dim=_CRITIC_DIM,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        num_atoms=5,
        device="cpu",
        use_compile=False,
        normalize_reward=False,
    )

    learner.update_critic(collected_batch)

    # The FlashSAC target receives the collector-encoded mask end to end:
    # dones keep the lifecycle semantics while truncated is bootstrap
    # eligibility, so clamp(1 - dones + truncated) yields the expected table.
    torch.testing.assert_close(captured["dones"], _EXPECTED_DONES)
    torch.testing.assert_close(captured["truncated"], _EXPECTED_TIMEOUT_BOOTSTRAP)
    bootstrap = torch.clamp(1.0 - captured["dones"] + captured["truncated"], 0.0, 1.0)
    torch.testing.assert_close(bootstrap, _EXPECTED_BOOTSTRAP)
    # Rewards arrive unmodified (no value bootstrap folded into the reward).
    torch.testing.assert_close(captured["reward"], _REWARDS)


def test_reward_normalizer_episode_carry_resets_on_every_done_flavor(
    collected_batch: dict[str, torch.Tensor],
) -> None:
    normalizer = RewardNormalizer(gamma=0.5, g_max=100.0, device=torch.device("cpu"))
    rewards = collected_batch["rewards"]
    dones = collected_batch["dones"]

    # One continuing step builds up the discounted-return carry.
    normalizer.update_from_transitions(torch.full((1, _NUM_ENVS), 2.0), torch.zeros(1, _NUM_ENVS))
    torch.testing.assert_close(normalizer.g_r, torch.full((_NUM_ENVS,), 2.0))

    # The replay terminal step: every done flavor (failure, timeout-only,
    # overlap) ends the episode carry, and the terminal transition reward is
    # retained as the start of the next episode's discounted return.
    normalizer.update_from_transitions(
        rewards.view(1, _NUM_ENVS),
        dones.view(1, _NUM_ENVS),
    )
    torch.testing.assert_close(
        normalizer.g_r,
        torch.tensor([0.5 * 2.0 + 1.0, 2.0, 3.0, 4.0]),
    )
