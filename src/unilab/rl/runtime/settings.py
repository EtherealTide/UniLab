"""Bounded tensor-runtime settings resolved before spawn.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.tensor_runtime``. This module is a leaf: it only defines the
bounds constants and the validated settings value object, and must stay
importable without Torch. Hydra config resolution lives in
:mod:`unilab.training.tensor_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_INFERENCE_SLOT_CAPACITY = 1
MAX_INFERENCE_SLOT_CAPACITY = 16
DEFAULT_COLLECTOR_METRICS_INTERVAL = 1
MAX_COLLECTOR_METRICS_INTERVAL = 10_000
DEFAULT_REPLAY_INGRESS_DEPTH = 2
MAX_REPLAY_INGRESS_DEPTH = 16


def _validated_positive_int(
    value: object,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a positive integer, got {value!r}")
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and no greater than {maximum}"
        raise ValueError(f"{label} must be a positive integer{suffix}, got {value!r}")
    return value


@dataclass(frozen=True)
class RuntimeSettings:
    """Effective, bounded tensor-runtime values resolved before spawn."""

    inference_slot_capacity: int
    collector_metrics_interval: int
    replay_ingress_depth: int
    replay_ingress_slot_rows: int
    batch_size: int
    updates_per_step: int
    num_envs: int
    configured_batch_size: int | None = None
    configured_updates_per_step: int | None = None
    configured_inference_slot_capacity: int | None = None
    configured_collector_metrics_interval: int | None = None
    configured_replay_ingress_depth: int | None = None
    configured_replay_ingress_slot_rows: int | None = None

    def __post_init__(self) -> None:
        num_envs = _validated_positive_int(self.num_envs, label="algo.num_envs")
        inference_capacity = _validated_positive_int(
            self.inference_slot_capacity,
            label="training.inference_slot_capacity",
            maximum=MAX_INFERENCE_SLOT_CAPACITY,
        )
        metrics_interval = _validated_positive_int(
            self.collector_metrics_interval,
            label="training.collector_metrics_interval",
            maximum=MAX_COLLECTOR_METRICS_INTERVAL,
        )
        ingress_depth = _validated_positive_int(
            self.replay_ingress_depth,
            label="training.replay_ingress_depth",
            maximum=MAX_REPLAY_INGRESS_DEPTH,
        )
        ingress_slot_rows = _validated_positive_int(
            self.replay_ingress_slot_rows,
            label="training.replay_ingress_slot_rows",
            maximum=num_envs,
        )
        batch_size = _validated_positive_int(self.batch_size, label="algo.batch_size")
        updates_per_step = _validated_positive_int(
            self.updates_per_step,
            label="algo.updates_per_step",
        )
        object.__setattr__(self, "inference_slot_capacity", inference_capacity)
        object.__setattr__(self, "collector_metrics_interval", metrics_interval)
        object.__setattr__(self, "replay_ingress_depth", ingress_depth)
        object.__setattr__(self, "replay_ingress_slot_rows", ingress_slot_rows)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "updates_per_step", updates_per_step)
        configured_values = (
            (
                "inference_slot_capacity",
                self.configured_inference_slot_capacity,
                inference_capacity,
            ),
            (
                "collector_metrics_interval",
                self.configured_collector_metrics_interval,
                metrics_interval,
            ),
            ("replay_ingress_depth", self.configured_replay_ingress_depth, ingress_depth),
            (
                "replay_ingress_slot_rows",
                self.configured_replay_ingress_slot_rows,
                ingress_slot_rows,
            ),
            ("batch_size", self.configured_batch_size, batch_size),
            ("updates_per_step", self.configured_updates_per_step, updates_per_step),
        )
        for name, configured, effective in configured_values:
            if configured is None:
                continue
            configured_int = _validated_positive_int(
                configured,
                label=f"configured {name}",
            )
            if configured_int != effective:
                raise ValueError(
                    f"configured {name} must match its effective value "
                    f"({configured_int!r} != {effective!r})"
                )

    @property
    def learner_sample_count(self) -> int:
        return self.batch_size * self.updates_per_step

    def manifest(self) -> dict[str, dict[str, object]]:
        return {
            "inference_ring_capacity": {
                "configured": self.configured_inference_slot_capacity,
                "default": DEFAULT_INFERENCE_SLOT_CAPACITY,
                "effective": self.inference_slot_capacity,
                "maximum": MAX_INFERENCE_SLOT_CAPACITY,
            },
            "collector_metrics_interval": {
                "configured": self.configured_collector_metrics_interval,
                "default": DEFAULT_COLLECTOR_METRICS_INTERVAL,
                "effective": self.collector_metrics_interval,
                "maximum": MAX_COLLECTOR_METRICS_INTERVAL,
            },
            "replay_ingress_depth": {
                "configured": self.configured_replay_ingress_depth,
                "default": DEFAULT_REPLAY_INGRESS_DEPTH,
                "effective": self.replay_ingress_depth,
                "maximum": MAX_REPLAY_INGRESS_DEPTH,
            },
            "replay_ingress_slot_rows": {
                "configured": self.configured_replay_ingress_slot_rows,
                "default": "algo.num_envs",
                "effective": self.replay_ingress_slot_rows,
                "maximum": self.num_envs,
            },
            "learner_sampling": {
                "configured_batch_size": self.configured_batch_size,
                "configured_updates_per_step": self.configured_updates_per_step,
                "configured_rows_per_sync": (
                    self.configured_batch_size * self.configured_updates_per_step
                    if (
                        self.configured_batch_size is not None
                        and self.configured_updates_per_step is not None
                    )
                    else None
                ),
                "default": "required owner configuration",
                "effective_batch_size": self.batch_size,
                "effective_updates_per_step": self.updates_per_step,
                "effective_rows_per_sync": self.learner_sample_count,
                "maximum": "device-memory budget",
            },
        }
