from __future__ import annotations

import multiprocessing as mp
import queue
from types import SimpleNamespace

import pytest
import torch

from unilab.rl.algos.appo.worker import (
    appo_collector_fn,
    compute_timeout_bootstrap_correction,
    put_latest_metrics,
)


def test_appo_collector_rejects_diagnostics_without_injected_guard_factory() -> None:
    class _Env:
        num_envs = 1
        play_capabilities = SimpleNamespace(supports_physics_state_playback=False)

        def set_nan_guard(self, guard: object) -> None:
            del guard
            raise AssertionError("no guard should be constructed without a factory")

    class _FakeRing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def attach_sync_primitives(self, *args: object) -> None:
            del args

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("unilab.rl.runtime.ipc.TensorRolloutRingBuffer", _FakeRing)
    monkeypatch.setattr(
        "unilab.rl.runtime.ipc.SharedWeightSync",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    try:
        with pytest.raises(ValueError, match="injected nan_guard_factory"):
            appo_collector_fn(
                stop_event=mp.Event(),
                env_factory=lambda num_envs, env_cfg_override=None: _Env(),
                rl_cfg={},
                num_envs=1,
                steps_per_env=1,
                shm_rollout_ring_buffer_name={},
                sync_primitives=(),
                obs_dim=1,
                action_dim=1,
                critic_dim=1,
                actor_weight_sync_name="",
                actor_weight_param_shapes={},
                critic_weight_sync_name="",
                critic_weight_param_shapes={},
                metrics_queue=queue.Queue(),
                nan_guard_cfg=SimpleNamespace(enabled=True),
            )
    finally:
        monkeypatch.undo()


class _FakeCritic:
    def __call__(self, obs):
        policy = obs["policy"]
        return policy.sum(dim=1, keepdim=True)


def test_compute_timeout_bootstrap_correction_uses_final_observation_value():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=torch.tensor([True, False]),
        final_critic=torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float32),
    )

    torch.testing.assert_close(correction, torch.tensor([2.5, 0.0], dtype=torch.float32))


def test_compute_timeout_bootstrap_correction_prefers_explicit_final_critic():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=torch.tensor([True, False]),
        final_critic=torch.tensor([[11.0, 13.0], [0.0, 0.0]], dtype=torch.float32),
    )

    torch.testing.assert_close(correction, torch.tensor([12.0, 0.0], dtype=torch.float32))


def test_put_latest_metrics_replaces_stale_item_when_queue_is_full(capsys):
    metrics_queue = queue.Queue(maxsize=1)
    metrics_queue.put_nowait({"total_steps": 1})

    put_latest_metrics(metrics_queue, {"total_steps": 2}, worker_name="APPOWorker")

    assert metrics_queue.get_nowait() == {"total_steps": 2}
    captured = capsys.readouterr()
    assert captured.err == ""
