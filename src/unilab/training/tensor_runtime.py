"""Hydra config resolution for the off-policy tensor-runtime contract.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.tensor_runtime``. Reading the owner ``DictConfig`` is training
assembly business, so these resolvers live here; the bounded constants and
value objects they produce live in :mod:`unilab.rl.runtime.settings` and
:mod:`unilab.rl.runtime.placement`.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

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
    TensorRuntimeSettings,
    _validated_positive_int,
)


def resolve_collector_tensor_native(
    cfg: DictConfig,
    *,
    device: str,
    algo_name: str,
) -> bool:
    """Compatibility resolver returning one placement contract's tensor mode."""
    return resolve_inference_transport(
        cfg, device=device, algo_name=algo_name
    ).collector_tensor_native


def resolve_inference_transport(
    cfg: DictConfig,
    *,
    device: str,
    algo_name: str,
    env_tensor_native: bool | None = None,
) -> InferencePlacement:
    """Resolve placement from the owner config's single config-source tree."""
    requested_transport = OmegaConf.select(cfg, "training.inference_transport", default=None)
    env_device = OmegaConf.select(cfg, "env.tensor_runtime_device", default=None)
    tensor_runtime = OmegaConf.select(cfg, "env.tensor_runtime", default=False)
    if tensor_runtime is None:
        tensor_runtime = False
    if type(tensor_runtime) is not bool:
        raise TypeError(
            f"{algo_name} env.tensor_runtime must be a boolean or omitted, got {tensor_runtime!r}"
        )
    if requested_transport is not None and not isinstance(requested_transport, str):
        raise TypeError(
            f"{algo_name} training.inference_transport must be a string or omitted, "
            f"got {requested_transport!r}"
        )
    if tensor_runtime and env_device is None:
        env_device = device
    if tensor_runtime and env_device is not None and not isinstance(env_device, str):
        raise TypeError(
            f"{algo_name} env.tensor_runtime_device must be a string or omitted, got {env_device!r}"
        )
    return resolve_inference_placement(
        learner_device=device,
        transport=requested_transport,
        tensor_runtime=tensor_runtime,
        tensor_runtime_device=env_device,
        env_tensor_native=env_tensor_native,
        algo_name=algo_name,
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


def _resolved_training_int(
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
    effective = _validated_positive_int(
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
    return _validated_positive_int(value, label=f"{algo_name} algo.{key}")


def resolve_tensor_runtime_settings(
    cfg: DictConfig,
    *,
    algo_name: str,
    num_envs: int,
) -> TensorRuntimeSettings:
    """Resolve all spawn-facing tensor-runtime knobs and learner sampling."""
    inference_capacity, configured_inference_capacity = _resolved_training_int(
        cfg,
        key="inference_slot_capacity",
        default=DEFAULT_INFERENCE_SLOT_CAPACITY,
        algo_name=algo_name,
        maximum=MAX_INFERENCE_SLOT_CAPACITY,
    )
    metrics_interval, configured_metrics_interval = _resolved_training_int(
        cfg,
        key="collector_metrics_interval",
        default=DEFAULT_COLLECTOR_METRICS_INTERVAL,
        algo_name=algo_name,
        maximum=MAX_COLLECTOR_METRICS_INTERVAL,
    )
    ingress_depth, configured_ingress_depth = _resolved_training_int(
        cfg,
        key="replay_ingress_depth",
        default=DEFAULT_REPLAY_INGRESS_DEPTH,
        algo_name=algo_name,
        maximum=MAX_REPLAY_INGRESS_DEPTH,
    )
    ingress_slot_rows, configured_ingress_slot_rows = _resolved_training_int(
        cfg,
        key="replay_ingress_slot_rows",
        default=num_envs,
        algo_name=algo_name,
    )
    batch_size = _required_algo_int(cfg, key="batch_size", algo_name=algo_name)
    updates_per_step = _required_algo_int(cfg, key="updates_per_step", algo_name=algo_name)
    return TensorRuntimeSettings(
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
