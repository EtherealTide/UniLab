"""Pure inference-placement resolution (owner: rl.runtime.placement).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/utils/test_tensor_runtime.py``; Hydra-level placement resolution
coverage lives in ``tests/training/test_tensor_runtime.py``.
"""

from __future__ import annotations

import pytest
import torch

from unilab.rl.runtime.placement import InferenceTransport, resolve_inference_placement


def test_collector_tensor_device_must_be_named_device() -> None:
    with pytest.raises(ValueError, match="collector tensor device must be one of"):
        resolve_inference_placement(
            learner_device="cuda:0",
            collector_tensor_device="accel",
            algo_name="SAC",
        )


def test_collector_tensor_device_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="collector tensor device must be a string or omitted"):
        resolve_inference_placement(
            learner_device="cuda:0",
            collector_tensor_device=1,
            algo_name="SAC",
        )


def test_collector_tensor_cuda_requires_cuda_learner() -> None:
    with pytest.raises(ValueError, match="collector_tensor_device=cuda requires a CUDA learner"):
        resolve_inference_placement(
            learner_device="cpu",
            collector_tensor_device="cuda",
            algo_name="SAC",
        )


def test_default_auto_follows_cuda_learner_to_cuda_ring() -> None:
    placement = resolve_inference_placement(learner_device="cuda:0", algo_name="SAC")

    assert placement.mode is InferenceTransport.CUDA
    assert placement.env_device == "cuda:0"
    assert placement.ring_device == "cuda:0"
    assert placement.learner_device == "cuda:0"
    assert placement.collector_tensor_native is True
    assert placement.staging_policy == "cuda_no_host_boundary"


def test_auto_follows_cpu_learner_to_host_ring() -> None:
    placement = resolve_inference_placement(
        learner_device="cpu",
        collector_tensor_device="auto",
        algo_name="SAC",
    )

    assert placement.mode is InferenceTransport.HOST
    assert placement.env_device == "cpu"
    assert placement.ring_device == "cpu"
    assert placement.collector_tensor_native is False
    assert placement.staging_policy == "cpu_no_device_transfer"


def test_explicit_cpu_with_cuda_learner_fails_closed() -> None:
    with pytest.raises(ValueError, match="collector_tensor_device=cpu with CUDA learner"):
        resolve_inference_placement(
            learner_device="cuda:1",
            collector_tensor_device="cpu",
            algo_name="SAC",
        )


def test_explicit_cpu_with_cuda_learner_requires_opt_in() -> None:
    placement = resolve_inference_placement(
        learner_device="cuda:1",
        collector_tensor_device="cpu",
        algo_name="SAC",
        allow_cpu_carrier_with_cuda_learner=True,
    )

    assert placement.mode is InferenceTransport.HOST
    assert placement.env_device == "cpu"
    assert placement.ring_device == "cpu"
    assert placement.learner_device == "cuda:1"
    assert placement.collector_tensor_native is False
    assert placement.staging_policy == "cpu_ring_explicit_learner_actor_h2d_action_d2h"


def test_explicit_cpu_with_cuda_learner_allowed_on_rocm_build(monkeypatch) -> None:
    """ROCm/HIP builds keep the CPU carrier as their only sharing mitigation."""
    monkeypatch.setattr(torch.version, "hip", "6.2")

    placement = resolve_inference_placement(
        learner_device="cuda:0",
        collector_tensor_device="cpu",
        algo_name="SAC",
    )

    assert placement.mode is InferenceTransport.HOST
    assert placement.staging_policy == "cpu_ring_explicit_learner_actor_h2d_action_d2h"


@pytest.mark.parametrize("learner_device", ["cpu", "mps"])
def test_explicit_cpu_stays_allowed_for_non_cuda_learners(learner_device: str) -> None:
    placement = resolve_inference_placement(
        learner_device=learner_device,
        collector_tensor_device="cpu",
        algo_name="SAC",
    )

    assert placement.mode is InferenceTransport.HOST
    assert placement.collector_tensor_native is False


def test_explicit_cuda_places_env_and_ring_on_learner_device() -> None:
    placement = resolve_inference_placement(
        learner_device="cuda:2",
        collector_tensor_device="cuda",
        algo_name="FlashSAC",
    )

    assert placement.mode is InferenceTransport.CUDA
    assert placement.env_device == "cuda:2"
    assert placement.ring_device == "cuda:2"
    assert placement.learner_device == "cuda:2"
    assert placement.collector_tensor_native is True
    assert placement.staging_policy == "cuda_no_host_boundary"
