"""Hydra config resolution for the tensor-runtime contract.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/utils/test_tensor_runtime.py``: reading the owner ``DictConfig`` is
training assembly business, so the resolver coverage lives here while the
value objects it produces are covered in ``tests/rl/runtime/``.
"""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from unilab.rl.runtime.placement import InferenceTransport
from unilab.rl.runtime.settings import (
    DEFAULT_COLLECTOR_METRICS_INTERVAL,
    DEFAULT_INFERENCE_SLOT_CAPACITY,
    DEFAULT_REPLAY_INGRESS_DEPTH,
    MAX_COLLECTOR_METRICS_INTERVAL,
    MAX_INFERENCE_SLOT_CAPACITY,
    MAX_REPLAY_INGRESS_DEPTH,
)
from unilab.training.tensor_runtime import (
    resolve_collector_metrics_interval,
    resolve_collector_tensor_placement,
    resolve_inference_slot_capacity,
    resolve_runtime_settings,
    select_collector_tensor_device,
)


def _cfg(
    training: dict | None = None,
    env: dict | None = None,
    algo: dict | None = None,
) -> Any:
    return OmegaConf.create(
        {
            "training": training or {},
            "env": env or {},
            "algo": algo or {"batch_size": 4, "updates_per_step": 2},
        }
    )


def test_runtime_settings_use_reviewed_defaults() -> None:
    settings = resolve_runtime_settings(_cfg(), algo_name="FlashSAC", num_envs=32)

    assert settings.inference_slot_capacity == DEFAULT_INFERENCE_SLOT_CAPACITY == 1
    assert settings.collector_metrics_interval == DEFAULT_COLLECTOR_METRICS_INTERVAL == 1
    assert settings.replay_ingress_depth == DEFAULT_REPLAY_INGRESS_DEPTH == 2
    assert settings.replay_ingress_slot_rows == 32
    assert settings.learner_sample_count == 8


def test_collector_tensor_device_selection() -> None:
    assert select_collector_tensor_device(_cfg()) == "auto"
    assert select_collector_tensor_device(_cfg({"collector_tensor_device": None})) == "auto"
    assert select_collector_tensor_device(_cfg({"collector_tensor_device": "cpu"})) == "cpu"
    assert select_collector_tensor_device(_cfg({"collector_tensor_device": "cuda"})) == "cuda"


def test_removed_training_keys_fail_closed() -> None:
    with pytest.raises(ValueError, match="training.sim_carrier_device was removed"):
        select_collector_tensor_device(_cfg({"sim_carrier_device": "cpu"}))

    with pytest.raises(ValueError, match="training.inference_transport was removed"):
        select_collector_tensor_device(_cfg({"inference_transport": "cuda"}))


def test_collector_tensor_device_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="training.collector_tensor_device must be a string"):
        select_collector_tensor_device(_cfg({"collector_tensor_device": 1}))


def test_runtime_settings_accept_reviewed_maxima() -> None:
    settings = resolve_runtime_settings(
        _cfg(
            {
                "inference_slot_capacity": MAX_INFERENCE_SLOT_CAPACITY,
                "collector_metrics_interval": MAX_COLLECTOR_METRICS_INTERVAL,
                "replay_ingress_depth": MAX_REPLAY_INGRESS_DEPTH,
                "replay_ingress_slot_rows": 3,
            }
        ),
        algo_name="SAC",
        num_envs=3,
    )

    assert settings.inference_slot_capacity == 16
    assert settings.collector_metrics_interval == 10_000
    assert settings.replay_ingress_depth == 16
    assert settings.replay_ingress_slot_rows == 3


def test_runtime_settings_accept_all_minima() -> None:
    settings = resolve_runtime_settings(
        _cfg(
            {
                "inference_slot_capacity": 1,
                "collector_metrics_interval": 1,
                "replay_ingress_depth": 1,
                "replay_ingress_slot_rows": 1,
            },
            algo={"batch_size": 1, "updates_per_step": 1},
        ),
        algo_name="FlashSAC",
        num_envs=1,
    )

    assert settings.inference_slot_capacity == 1
    assert settings.collector_metrics_interval == 1
    assert settings.replay_ingress_depth == 1
    assert settings.replay_ingress_slot_rows == 1
    assert settings.batch_size == 1
    assert settings.updates_per_step == 1
    assert settings.num_envs == 1
    assert settings.learner_sample_count == 1


@pytest.mark.parametrize(
    ("key", "maximum"),
    [
        ("inference_slot_capacity", MAX_INFERENCE_SLOT_CAPACITY),
        ("collector_metrics_interval", MAX_COLLECTOR_METRICS_INTERVAL),
        ("replay_ingress_depth", MAX_REPLAY_INGRESS_DEPTH),
    ],
)
def test_runtime_settings_reject_values_above_maxima(key: str, maximum: int) -> None:
    with pytest.raises(ValueError, match=f"no greater than {maximum}"):
        resolve_runtime_settings(
            _cfg({key: maximum + 1}),
            algo_name="SAC",
            num_envs=4,
        )


@pytest.mark.parametrize(
    "key",
    [
        "inference_slot_capacity",
        "collector_metrics_interval",
        "replay_ingress_depth",
        "replay_ingress_slot_rows",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, "2", 1.0])
def test_runtime_settings_reject_invalid_training_values(key: str, value: Any) -> None:
    error = TypeError if type(value) is not int else ValueError
    with pytest.raises(error, match="training\\." + key):
        resolve_runtime_settings(
            _cfg({key: value}),
            algo_name="FlashSAC",
            num_envs=4,
        )


@pytest.mark.parametrize("key", ["batch_size", "updates_per_step", "num_envs"])
@pytest.mark.parametrize("value", [0, -1, True, "2", 1.0])
def test_runtime_settings_reject_invalid_algo_values(key: str, value: Any) -> None:
    error = TypeError if type(value) is not int else ValueError
    algo = {"batch_size": 4, "updates_per_step": 2}
    num_envs = 4
    if key == "num_envs":
        num_envs = value
        training = {"replay_ingress_slot_rows": 2}
    else:
        algo[key] = value
        training = None
    with pytest.raises(error, match="algo\\." + key):
        resolve_runtime_settings(
            _cfg(training=training, algo=algo),
            algo_name="FlashSAC",
            num_envs=num_envs,
        )


@pytest.mark.parametrize("key", ["batch_size", "updates_per_step"])
def test_runtime_settings_require_learner_sampling(key: str) -> None:
    algo = {"batch_size": 4, "updates_per_step": 2}
    del algo[key]
    with pytest.raises(ValueError, match=f"FlashSAC algo\\.{key} must be configured"):
        resolve_runtime_settings(_cfg(algo=algo), algo_name="FlashSAC", num_envs=4)


def test_replay_ingress_slot_rows_may_be_smaller_than_num_envs() -> None:
    settings = resolve_runtime_settings(
        _cfg({"replay_ingress_slot_rows": 2}),
        algo_name="SAC",
        num_envs=16,
    )

    assert settings.replay_ingress_slot_rows == 2


def test_replay_ingress_slot_rows_reject_rows_above_num_envs() -> None:
    with pytest.raises(ValueError, match="no greater than 4"):
        resolve_runtime_settings(
            _cfg({"replay_ingress_slot_rows": 5}),
            algo_name="SAC",
            num_envs=4,
        )


def test_tensor_runtime_manifest_records_configured_effective_and_bounds() -> None:
    settings = resolve_runtime_settings(
        _cfg(
            {
                "inference_slot_capacity": 2,
                "collector_metrics_interval": 100,
                "replay_ingress_depth": 3,
                "replay_ingress_slot_rows": 4,
            },
            algo={"batch_size": 5, "updates_per_step": 3},
        ),
        algo_name="FlashSAC",
        num_envs=8,
    )
    manifest = settings.manifest()

    assert manifest["inference_ring_capacity"] == {
        "configured": 2,
        "default": 1,
        "effective": 2,
        "maximum": 16,
    }
    assert manifest["collector_metrics_interval"]["maximum"] == 10_000
    assert manifest["replay_ingress_depth"]["configured"] == 3
    assert manifest["replay_ingress_slot_rows"]["maximum"] == 8
    assert manifest["learner_sampling"] == {
        "configured_batch_size": 5,
        "configured_updates_per_step": 3,
        "configured_rows_per_sync": 15,
        "default": "required owner configuration",
        "effective_batch_size": 5,
        "effective_updates_per_step": 3,
        "effective_rows_per_sync": 15,
        "maximum": "device-memory budget",
    }


@pytest.mark.parametrize(
    "resolver",
    [resolve_inference_slot_capacity, resolve_collector_metrics_interval],
)
def test_tensor_runtime_intervals_default_to_one(resolver) -> None:
    cfg = _cfg()

    assert resolver(cfg, algo_name="SAC") == 1
    assert resolver(_cfg({"inference_slot_capacity": None}), algo_name="SAC") == 1


def test_removed_env_tensor_runtime_fields_fail_closed() -> None:
    for env in (
        {"tensor_runtime": True},
        {"tensor_runtime": None},
        {"tensor_runtime_device": None},
    ):
        with pytest.raises(ValueError, match="env.tensor_runtime"):
            resolve_collector_tensor_placement(_cfg(env=env), device="cuda", algo_name="SAC")


def test_removed_training_keys_fail_closed_in_placement_resolution() -> None:
    for key in ("sim_carrier_device", "inference_transport"):
        with pytest.raises(ValueError, match=f"training.{key} was removed"):
            resolve_collector_tensor_placement(
                _cfg(training={key: "cuda"}),
                device="cuda:0",
                algo_name="SAC",
            )


def test_default_auto_follows_cuda_learner_to_cuda_ring() -> None:
    placement = resolve_collector_tensor_placement(_cfg(), device="cuda:3", algo_name="SAC")

    assert placement.mode is InferenceTransport.CUDA
    assert placement.env_device == "cuda:3"
    assert placement.ring_device == "cuda:3"
    assert placement.learner_device == "cuda:3"
    assert placement.collector_tensor_native is True
    assert placement.staging_policy == "cuda_no_host_boundary"
    assert placement.manifest() == {
        "mode": "cuda",
        "env_device": "cuda:3",
        "ring_device": "cuda:3",
        "learner_device": "cuda:3",
        "staging_policy": "cuda_no_host_boundary",
        "transport_table": {
            "ring_device_type": "cuda",
            "requires_tensor_native": True,
            "requires_rank_local_device_match": True,
            "publication_sync": "cuda_ipc_events",
            "evidence": "policy",
        },
    }


def test_host_transport_labels_non_cuda_accelerator_learners_with_device_copy() -> None:
    # torch.mps (Apple Metal Performance Shaders) learners still cross devices
    # at the CPU-ring copy_ boundary; they must not claim cpu_no_device_transfer.
    placement = resolve_collector_tensor_placement(_cfg(), device="mps", algo_name="SAC")

    assert placement.mode is InferenceTransport.HOST
    assert placement.env_device == "cpu"
    assert placement.ring_device == "cpu"
    assert placement.learner_device == "mps"
    assert placement.collector_tensor_native is False
    assert placement.staging_policy == "cpu_ring_explicit_learner_device_copy"


def test_host_transport_cpu_learner_reports_no_device_transfer() -> None:
    placement = resolve_collector_tensor_placement(_cfg(), device="cpu", algo_name="SAC")

    assert placement.mode is InferenceTransport.HOST
    assert placement.staging_policy == "cpu_no_device_transfer"


def test_explicit_collector_tensor_cuda_requests_cuda_ring() -> None:
    placement = resolve_collector_tensor_placement(
        _cfg(training={"collector_tensor_device": "cuda"}),
        device="cuda:4",
        algo_name="FlashSAC",
    )

    assert placement.mode is InferenceTransport.CUDA
    assert placement.env_device == "cuda:4"
    assert placement.ring_device == "cuda:4"
    assert placement.collector_tensor_native is True


def test_explicit_collector_tensor_cpu_with_cuda_learner_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER", raising=False)
    with pytest.raises(ValueError, match="collector_tensor_device=cpu with CUDA learner"):
        resolve_collector_tensor_placement(
            _cfg(training={"collector_tensor_device": "cpu"}),
            device="cuda:0",
            algo_name="SAC",
        )


def test_explicit_collector_tensor_cpu_opt_in_keeps_cpu_ring_with_cuda_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attribution-baseline/debug opt-in retains the pre-#2483 topology."""
    monkeypatch.setenv("UNILAB_ALLOW_CUDA_LEARNER_CPU_CARRIER", "1")
    placement = resolve_collector_tensor_placement(
        _cfg(training={"collector_tensor_device": "cpu"}),
        device="cuda:0",
        algo_name="SAC",
    )

    assert placement.mode is InferenceTransport.HOST
    assert placement.env_device == "cpu"
    assert placement.ring_device == "cpu"
    assert placement.collector_tensor_native is False
    assert placement.staging_policy == "cpu_ring_explicit_learner_actor_h2d_action_d2h"


@pytest.mark.parametrize("device", [None, "auto", "cuda"])
def test_collector_tensor_device_owner_yaml_semantics(device: str | None) -> None:
    """The owner enum is exactly auto/cpu/cuda with null retained for auto.

    The ``cpu`` member's CUDA-learner pairing is covered by the fail-closed
    and opt-in tests above; every other value resolves the CUDA ring here.
    """
    placement = resolve_collector_tensor_placement(
        _cfg(training={"collector_tensor_device": device}),
        device="cuda:0",
        algo_name="SAC",
    )

    assert placement.mode is InferenceTransport.CUDA


def test_collector_tensor_cuda_requires_cuda_learner() -> None:
    with pytest.raises(ValueError, match="collector_tensor_device=cuda requires a CUDA learner"):
        resolve_collector_tensor_placement(
            _cfg(training={"collector_tensor_device": "cuda"}),
            device="cpu",
            algo_name="SAC",
        )


def test_removed_host_transport_spelling_fails_closed() -> None:
    with pytest.raises(ValueError, match="collector tensor device must be one of"):
        resolve_collector_tensor_placement(
            _cfg(training={"collector_tensor_device": "host"}),
            device="cuda:0",
            algo_name="SAC",
        )
