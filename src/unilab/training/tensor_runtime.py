"""Hydra config resolution for the off-policy tensor-runtime contract.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.tensor_runtime``. Reading the owner ``DictConfig`` is training
assembly business, so these resolvers live here; the bounded constants and
value objects they produce live in :mod:`unilab.rl.runtime.settings` and
:mod:`unilab.rl.runtime.placement`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from unilab.base.process_device import (
    canonicalize_collector_tensor_device,
    is_host_bridge_backend,
)
from unilab.process_context import cuda_learner_cpu_carrier_opted_in
from unilab.rl.runtime.placement import (
    InferencePlacement,
    resolve_inference_placement,
)
from unilab.rl.runtime.settings import (
    DEFAULT_COLLECTOR_METRICS_INTERVAL,
    DEFAULT_INFERENCE_SLOT_CAPACITY,
    DEFAULT_REPLAY_INGRESS_DEPTH,
    MAX_COLLECTOR_METRICS_INTERVAL,
    MAX_INFERENCE_SLOT_CAPACITY,
    MAX_REPLAY_INGRESS_DEPTH,
    RuntimeSettings,
    validated_positive_int,
)


def select_collector_tensor_device(cfg: DictConfig) -> str:
    """Select the collector-tensor device request and reject removed keys."""
    training = OmegaConf.select(cfg, "training", default=None)
    if training is not None:
        if "sim_carrier_device" in training:
            raise ValueError(
                "training.sim_carrier_device was removed; use "
                "training.collector_tensor_device (accepted values: auto, cpu, or cuda)"
            )
        if "inference_transport" in training:
            raise ValueError(
                "training.inference_transport was removed; the inference transport is "
                "derived from training.collector_tensor_device "
                "(accepted values: auto, cpu, or cuda)"
            )
    value = OmegaConf.select(cfg, "training.collector_tensor_device", default="auto")
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise TypeError(
            f"training.collector_tensor_device must be a string or omitted, got {value!r}"
        )
    return value


def resolve_recorded_collector_tensor_device(
    sim_backend: str,
    env_cfg_override: Mapping[str, Any] | None,
) -> str | None:
    """Resolve the collector-tensor device recorded in ``run_config.json``.

    Every HOST_BRIDGE backend routes ``training.collector_tensor_device``;
    gating the record on one backend name drops the resolved device from the
    other HOST_BRIDGE backends' run configs.
    """

    if not is_host_bridge_backend(sim_backend):
        return None
    return canonicalize_collector_tensor_device(
        (env_cfg_override or {}).get("collector_tensor_device")
    )


def resolve_collector_tensor_placement(
    cfg: DictConfig,
    *,
    device: str,
    algo_name: str,
) -> InferencePlacement:
    """Resolve placement from the owner config's single config-source tree."""
    env_cfg = OmegaConf.select(cfg, "env", default=None)
    if env_cfg is not None and ("tensor_runtime" in env_cfg or "tensor_runtime_device" in env_cfg):
        raise ValueError(
            "env.tensor_runtime and env.tensor_runtime_device were removed; "
            "tensor-native inference follows training.collector_tensor_device"
        )
    return resolve_inference_placement(
        learner_device=device,
        collector_tensor_device=select_collector_tensor_device(cfg),
        algo_name=algo_name,
        allow_cpu_carrier_with_cuda_learner=cuda_learner_cpu_carrier_opted_in(),
    )


def _resolve_positive_training_int(
    cfg: DictConfig,
    *,
    key: str,
    default: int,
    algo_name: str,
    maximum: int | None = None,
) -> int:
    value = OmegaConf.select(cfg, f"training.{key}", default=default)
    if value is None:
        return default
    if type(value) is not int:
        raise TypeError(
            f"{algo_name} training.{key} must be a positive integer or omitted, got {value!r}"
        )
    suffix = "" if maximum is None else f" and no greater than {maximum}"
    if value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(
            f"{algo_name} training.{key} must be a positive integer{suffix}, got {value!r}"
        )
    return value


def resolve_inference_slot_capacity(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the bounded inference-ring capacity before IPC construction."""
    return _resolve_positive_training_int(
        cfg,
        key="inference_slot_capacity",
        default=DEFAULT_INFERENCE_SLOT_CAPACITY,
        algo_name=algo_name,
        maximum=MAX_INFERENCE_SLOT_CAPACITY,
    )


def resolve_collector_metrics_interval(cfg: DictConfig, *, algo_name: str) -> int:
    """Resolve the device-metric compaction/reporting interval."""
    return _resolve_positive_training_int(
        cfg,
        key="collector_metrics_interval",
        default=DEFAULT_COLLECTOR_METRICS_INTERVAL,
        algo_name=algo_name,
        maximum=MAX_COLLECTOR_METRICS_INTERVAL,
    )


def _resolve_configured_training_int(
    cfg: DictConfig,
    *,
    key: str,
    default: int,
    algo_name: str,
    maximum: int | None = None,
) -> tuple[int, int | None]:
    value = OmegaConf.select(cfg, f"training.{key}", default=default)
    if value is None:
        return default, None
    effective = validated_positive_int(
        value,
        label=f"{algo_name} training.{key}",
        maximum=maximum,
    )
    return effective, effective


def _required_algo_int(cfg: DictConfig, *, key: str, algo_name: str) -> int:
    value = OmegaConf.select(cfg, f"algo.{key}")
    if value is None:
        raise ValueError(
            f"{algo_name} algo.{key} must be configured as a positive integer, got None"
        )
    return validated_positive_int(value, label=f"{algo_name} algo.{key}")


def resolve_runtime_settings(
    cfg: DictConfig,
    *,
    algo_name: str,
    num_envs: int,
) -> RuntimeSettings:
    """Resolve all spawn-facing tensor-runtime knobs and learner sampling."""
    inference_capacity, configured_inference_capacity = _resolve_configured_training_int(
        cfg,
        key="inference_slot_capacity",
        default=DEFAULT_INFERENCE_SLOT_CAPACITY,
        algo_name=algo_name,
        maximum=MAX_INFERENCE_SLOT_CAPACITY,
    )
    metrics_interval, configured_metrics_interval = _resolve_configured_training_int(
        cfg,
        key="collector_metrics_interval",
        default=DEFAULT_COLLECTOR_METRICS_INTERVAL,
        algo_name=algo_name,
        maximum=MAX_COLLECTOR_METRICS_INTERVAL,
    )
    ingress_depth, configured_ingress_depth = _resolve_configured_training_int(
        cfg,
        key="replay_ingress_depth",
        default=DEFAULT_REPLAY_INGRESS_DEPTH,
        algo_name=algo_name,
        maximum=MAX_REPLAY_INGRESS_DEPTH,
    )
    ingress_slot_rows, configured_ingress_slot_rows = _resolve_configured_training_int(
        cfg,
        key="replay_ingress_slot_rows",
        default=num_envs,
        algo_name=algo_name,
    )
    batch_size = _required_algo_int(cfg, key="batch_size", algo_name=algo_name)
    updates_per_step = _required_algo_int(cfg, key="updates_per_step", algo_name=algo_name)
    return RuntimeSettings(
        inference_slot_capacity=inference_capacity,
        collector_metrics_interval=metrics_interval,
        replay_ingress_depth=ingress_depth,
        replay_ingress_slot_rows=ingress_slot_rows,
        batch_size=batch_size,
        updates_per_step=updates_per_step,
        num_envs=num_envs,
        configured_batch_size=batch_size,
        configured_updates_per_step=updates_per_step,
        configured_inference_slot_capacity=configured_inference_capacity,
        configured_collector_metrics_interval=configured_metrics_interval,
        configured_replay_ingress_depth=configured_ingress_depth,
        configured_replay_ingress_slot_rows=configured_ingress_slot_rows,
    )
