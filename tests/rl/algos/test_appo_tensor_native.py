"""Strict tensor-native APPO integration and Gaussian metric tests."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from tests.rl._fakes import StrictTensorEnv, StrictTensorEnvFactory
from unilab.rl.algos.appo.learner import APPOLearner, gaussian_distribution_std
from unilab.rl.algos.appo.runner import APPORunner

pytestmark = pytest.mark.slow


def _actor(obs_dim: int, action_dim: int) -> MLPModel:
    obs = torch.zeros((1, obs_dim))
    td = TensorDict({"policy": obs}, batch_size=1)
    return MLPModel(
        td,
        {"actor": ["policy"]},
        "actor",
        action_dim,
        hidden_dims=[8],
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
    )


def _critic(obs_dim: int) -> MLPModel:
    obs = torch.zeros((1, obs_dim))
    td = TensorDict({"policy": obs}, batch_size=1)
    return MLPModel(td, {"critic": ["policy"]}, "critic", 1, hidden_dims=[8])


def test_appo_runner_completes_two_iterations_without_numpy_env_carrier() -> None:
    num_envs, steps, obs_dim, action_dim = 4, 2, 3, 2
    rl_cfg = {
        "seed": 7,
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [8],
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [8]},
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "enable_compile": False,
        },
    }
    runner = APPORunner(
        env_name="StrictTensorEnv",
        env_factory=StrictTensorEnvFactory(obs_dim, action_dim),
        env_cfg_overrides={},
        rl_cfg=rl_cfg,
        device="cpu",
        collector_device="cpu",
        num_envs=num_envs,
        steps_per_env=steps,
        replay_queue_size=1,
    )
    # Ensure dimensions and actor construction align with the strict fixture.
    assert runner.obs_dim == obs_dim
    assert runner.critic_dim == obs_dim + 1
    assert runner.action_dim == action_dim

    with tempfile.TemporaryDirectory() as tmpdir:
        runner.learn(max_iterations=2, save_interval=0, log_dir=tmpdir)
    runner.close()


def test_appo_learner_std_metric_does_not_require_sample_cache() -> None:
    torch.manual_seed(11)
    actor = _actor(3, 2)
    learner = APPOLearner(
        actor=actor,
        critic=_critic(4),
        num_learning_epochs=1,
        num_mini_batches=2,
        device="cpu",
        enable_compile=False,
    )
    batch = learner.process_batch(
        {
            "observations": torch.randn(4, 3, 3),
            "actions": torch.randn(4, 3, 2),
            "actions_log_prob": torch.randn(4, 3),
            "rewards": torch.randn(4, 3),
            "dones": torch.zeros(4, 3),
            "last_obs": torch.randn(3, 3),
            "critic": torch.randn(4, 3, 4),
            "last_critic": torch.randn(3, 4),
        }
    )
    # Deliberately do not invoke actor.forward before update: metrics must come
    # from distribution.std_param, not GaussianDistribution._distribution.
    assert actor.distribution._distribution is None
    metrics = learner.update(batch)
    assert metrics["Policy/mean_std"] == pytest.approx(0.5, rel=0.02)
    assert actor.distribution._distribution is None


def test_gaussian_distribution_std_handles_scalar_and_log_parameterization() -> None:
    scalar = _actor(3, 2)
    assert torch.allclose(gaussian_distribution_std(scalar), scalar.distribution.std_param)

    log_actor = MLPModel(
        TensorDict({"policy": torch.zeros((1, 3))}, batch_size=1),
        {"actor": ["policy"]},
        "actor",
        2,
        hidden_dims=[8],
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.25,
            "std_type": "log",
        },
    )
    expected = torch.exp(log_actor.distribution.log_std_param)
    assert torch.allclose(gaussian_distribution_std(log_actor), expected)


def test_timeout_bootstrap_correction_uses_final_critic_value() -> None:
    from unilab.rl.algos.appo.worker import compute_timeout_bootstrap_correction

    class _Critic:
        def __call__(self, obs):
            return obs["policy"].sum(dim=1, keepdim=True)

    correction = compute_timeout_bootstrap_correction(
        critic=_Critic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=torch.tensor([True, False]),
        final_critic=torch.tensor([[2.0, 3.0], [9.0, 9.0]]),
    )
    torch.testing.assert_close(correction, torch.tensor([2.5, 0.0]))


def test_strict_tensor_env_rejects_numpy_actions() -> None:
    env = StrictTensorEnv()
    with pytest.raises(TypeError, match="action must be torch.Tensor"):
        env.step(np.zeros((1, 2), dtype=np.float32))


def test_strict_tensor_env_rejects_numpy_reset_indices() -> None:
    env = StrictTensorEnv()
    with pytest.raises(TypeError, match="reset indices must be torch.Tensor"):
        env.reset(np.array([0], dtype=np.int64))


def test_timeout_bootstrap_correction_supports_cross_device_final_critic() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from unilab.rl.algos.appo.worker import compute_timeout_bootstrap_correction

    class _Critic:
        def __call__(self, obs):
            return obs["policy"].sum(dim=1, keepdim=True)

    cpu_final = torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float32)
    cuda_mask = torch.tensor([True, False], device="cuda")

    correction = compute_timeout_bootstrap_correction(
        critic=_Critic(),
        collector_device="cuda",
        gamma=0.5,
        timeout_mask=cuda_mask,
        final_critic=cpu_final,
    )

    assert correction.device.type == "cuda"
    torch.testing.assert_close(correction, torch.tensor([2.5, 0.0], device="cuda"))
