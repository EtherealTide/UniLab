from __future__ import annotations

import numpy as np
import torch

from unilab.managers._noise.noise_cfg import UniformNoiseCfg
from unilab.tasks.motion_tracking.common.tensor_runtime import (
    TensorEpisodeMetrics,
    TensorObservationNoise,
    TensorResetPlan,
    semantic_fingerprint,
)
from unilab.tasks.motion_tracking.common.tensor_state_store import TensorDeviceStateStore


def test_semantic_fingerprint_rejects_semantic_mutation() -> None:
    base = {"std": 0.2, "enabled": True}
    mutated = dict(base)
    mutated["std"] = 0.25

    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.owner.v1", mutated
    )
    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.other.v1", base
    )


def test_tensor_observation_noise_uses_declared_uniform_bounds() -> None:
    noise = TensorObservationNoise.from_uniform_terms(
        (
            UniformNoiseCfg(n_min=-0.1, n_max=0.2),
            UniformNoiseCfg(n_min=-2.0, n_max=-1.0),
        ),
        (2, 3),
        device=torch.device("cpu"),
    )
    observations = torch.zeros((4, 6))
    generator = torch.Generator().manual_seed(17)

    corrupted = noise.apply(observations, cursor=1, generator=generator)

    assert corrupted is observations
    assert bool((corrupted[:, 1:3] >= -0.1).all()) and bool((corrupted[:, 1:3] <= 0.2).all())
    assert bool((corrupted[:, 3:6] >= -2.0).all()) and bool((corrupted[:, 3:6] <= -1.0).all())
    torch.testing.assert_close(corrupted[:, 0], torch.zeros(4))


def test_tensor_reset_plan_caches_selected_rows() -> None:
    plan = TensorResetPlan(
        terminated=torch.tensor([False, True, False, True]),
        truncated=torch.tensor([True, False, False, False]),
    )

    rows = plan.rows

    torch.testing.assert_close(rows, torch.tensor([0, 1, 3]))
    assert plan.rows is rows


def test_tensor_episode_metrics_include_terminal_then_reset() -> None:
    metrics = TensorEpisodeMetrics.create(3, torch.device("cpu"))

    metrics.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False]))
    finished = metrics.finished_values(torch.tensor([1]))
    metrics.reset(torch.tensor([1]))
    metrics.update(torch.tensor([4.0, 5.0, 6.0]), torch.tensor([False, False, True]))

    torch.testing.assert_close(finished, torch.tensor([[2.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 5.0, 9.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 1, 2]))
    metrics.reset(torch.tensor([1, 2]))
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 0, 0]))


class _HostBridgeBackend:
    backend_type = "mujoco"

    def get_state_views(self, names, device=None):
        assert names == ("qpos", "qvel")
        return {
            "qpos": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]],
                device=device,
            ),
            "qvel": torch.zeros((1, 7), device=device),
        }

    def get_sensor_view(self, name, device=None):
        values = {"pelvis_local_linvel": (1.0, 2.0, 3.0), "torso_gyro": (-1.0, -2.0, -3.0)}
        return torch.tensor(values[name], device=device).unsqueeze(0)

    def get_body_pos_w(self, body_ids):
        return np.array([[[0.1, 0.2, 0.3]]], dtype=np.float64)

    def get_body_quat_w(self, body_ids):
        return np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64)

    def get_body_lin_vel_w(self, body_ids):
        return np.zeros((1, 1, 3), dtype=np.float64)

    def get_body_ang_vel_w(self, body_ids):
        return np.zeros((1, 1, 3), dtype=np.float64)


def test_tensor_state_store_full_read_validates_layout_and_finite_state() -> None:
    store = TensorDeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=np.array([7], dtype=np.int64),
        joint_qvel_ids=np.array([6], dtype=np.int64),
        body_names=("pelvis",),
        body_ids=np.array([0], dtype=np.intp),
    )

    store.read()
    store.validate_finite()

    torch.testing.assert_close(store.joint_pos, torch.tensor([[0.25]]))
    torch.testing.assert_close(store.linvel, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(store.robot_body_pos, torch.tensor([[[0.1, 0.2, 0.3]]]))
