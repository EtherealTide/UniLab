from __future__ import annotations

import numpy as np
import pytest
import torch
from scripts.benchmark.torch_env.motion_tracking import (
    ANCHOR_IDX,
    N_BINS,
    SAMPLER_ALPHA,
    MotionTrackingWorkload,
)
from scripts.benchmark.torch_env.xp import TorchBackend, TorchRng


def _align_to_first_frame(workload: MotionTrackingWorkload) -> None:
    workload.current_frames.zero_()
    workload.body_pos[:] = workload.m_body_pos[0]
    workload.body_quat[:] = workload.m_body_quat[0]


def test_motion_tracking_failure_accumulator_stays_host_free_and_faithful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TorchBackend("cpu")
    workload = MotionTrackingWorkload(
        backend, TorchRng("cpu"), seed=7, vectorized_reset_rng=True, num_envs=4
    )
    initial_bins = torch.linspace(0.1, 0.4, N_BINS, dtype=torch.float32)
    workload.bin_failed_count.copy_(initial_bins)
    _align_to_first_frame(workload)

    def fail_on_host_sync(_) -> bool:
        raise AssertionError("adaptive sampler must not synchronize a scalar any()")

    monkeypatch.setattr(backend, "any_scalar", fail_on_host_sync)

    _, _, terminated = workload.update_state(should_log=False)
    assert not bool(terminated.any())
    torch.testing.assert_close(workload.bin_failed_count, initial_bins)

    _align_to_first_frame(workload)
    workload.body_pos[0, ANCHOR_IDX, 2] += 1.0
    before_failure = workload.bin_failed_count.clone()
    _, _, terminated = workload.update_state(should_log=False)

    assert bool(terminated[0]) and not bool(terminated[1:].any())
    expected = np.zeros(N_BINS, dtype=np.float32)
    expected[0] = 1.0
    expected = (SAMPLER_ALPHA * expected + (1.0 - SAMPLER_ALPHA) * before_failure.numpy()).astype(
        np.float32
    )
    torch.testing.assert_close(workload.bin_failed_count, torch.from_numpy(expected))
