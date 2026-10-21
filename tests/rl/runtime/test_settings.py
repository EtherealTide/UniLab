"""Bounded tensor-runtime settings value object (owner: rl.runtime.settings).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/utils/test_tensor_runtime.py``; Hydra config resolution coverage lives
in ``tests/training/test_tensor_runtime.py``.
"""

from __future__ import annotations

import pytest

from unilab.rl.runtime.settings import (
    DEFAULT_COLLECTOR_METRICS_INTERVAL,
    DEFAULT_INFERENCE_SLOT_CAPACITY,
    DEFAULT_REPLAY_INGRESS_DEPTH,
    MAX_COLLECTOR_METRICS_INTERVAL,
    MAX_INFERENCE_SLOT_CAPACITY,
    MAX_REPLAY_INGRESS_DEPTH,
    RuntimeSettings,
)


def test_runtime_bounds_constants_match_reviewed_values() -> None:
    assert DEFAULT_INFERENCE_SLOT_CAPACITY == 1
    assert MAX_INFERENCE_SLOT_CAPACITY == 16
    assert DEFAULT_COLLECTOR_METRICS_INTERVAL == 1
    assert MAX_COLLECTOR_METRICS_INTERVAL == 10_000
    assert DEFAULT_REPLAY_INGRESS_DEPTH == 2
    assert MAX_REPLAY_INGRESS_DEPTH == 16


def test_direct_settings_object_validates_all_spawn_facing_values() -> None:
    with pytest.raises(ValueError, match="training.replay_ingress_slot_rows"):
        RuntimeSettings(
            inference_slot_capacity=1,
            collector_metrics_interval=1,
            replay_ingress_depth=2,
            replay_ingress_slot_rows=5,
            batch_size=4,
            updates_per_step=2,
            num_envs=4,
        )


def test_direct_settings_object_rejects_inconsistent_configured_evidence() -> None:
    with pytest.raises(ValueError, match="configured replay_ingress_depth must match"):
        RuntimeSettings(
            inference_slot_capacity=1,
            collector_metrics_interval=1,
            replay_ingress_depth=2,
            configured_replay_ingress_depth=3,
            replay_ingress_slot_rows=1,
            batch_size=1,
            updates_per_step=1,
            num_envs=1,
        )
