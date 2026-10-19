from __future__ import annotations

import torch
from scripts.benchmark.torch_env.motion_tracking import MotionTrackingWorkload
from scripts.benchmark.torch_env.walk_flat import WalkFlatWorkload
from scripts.benchmark.torch_env.xp import TorchBackend, TorchRng


def _fill_obs(workload) -> None:
    for key in ("obs", "critic"):
        workload.obs[key] = (
            torch.linspace(
                1.0, float(workload.n_envs), workload.n_envs, dtype=workload.obs[key].dtype
            )[:, None]
            .expand_as(workload.obs[key])
            .clone()
        )


def _assert_terminal_observation(workload, env_ids: torch.Tensor) -> None:
    for key in ("obs", "critic"):
        expected = (env_ids.to(workload.final_observation[key].dtype) + 1)[:, None].expand_as(
            workload.final_observation[key][env_ids]
        )
        torch.testing.assert_close(workload.final_observation[key][env_ids], expected)

    assert not any(name.startswith("compat_") for name in vars(workload))


def test_walk_flat_reset_done_uses_single_final_observation() -> None:
    workload = WalkFlatWorkload(TorchBackend("cpu"), TorchRng("cpu"), seed=7, num_envs=4)
    _fill_obs(workload)
    env_ids = torch.tensor([0, 2], dtype=torch.int64)

    workload.reset_done(env_ids)

    _assert_terminal_observation(workload, env_ids)


def test_motion_tracking_reset_done_uses_single_final_observation() -> None:
    workload = MotionTrackingWorkload(
        TorchBackend("cpu"), TorchRng("cpu"), seed=7, vectorized_reset_rng=True, num_envs=4
    )
    _fill_obs(workload)
    env_ids = torch.tensor([1, 3], dtype=torch.int64)

    workload.reset_done(env_ids)

    _assert_terminal_observation(workload, env_ids)
