"""Learner-side custom actor contract for FlashSAC.

Downstream packages replace the built-in :class:`FlashSACActor` with their own
policy while reusing the FlashSAC learner, optimizers, DP gradient sync, and
off-policy runtime. Construction is declared in the owner Hydra config:

.. code-block:: yaml

    algo:
      algo_params:
        actor_config:
          builder: "some_package.actors:build_actor"
          kwargs: {hidden_dim: 256}

The builder is a dotted import path (``pkg.module:callable`` or
``pkg.module.callable``) resolved with :func:`importlib.import_module`, so the
downstream module only needs to be importable in the training process (and in
any spawn subprocess that rebuilds the actor). ``kwargs`` must be
JSON-serializable: they are persisted verbatim in checkpoints.

The builder is called as ``builder(obs_dim=..., action_dim=..., device=...,
**kwargs)`` and must return an ``nn.Module`` implementing the custom actor
contract:

- ``forward(observations, training) -> (actions, info)``: training forward.
  ``actions`` is the post-transform action ``(B, action_dim)``; ``info`` maps
  ``"log_prob"`` to the log-probability of the *returned* action ``(B,)``,
  including the action transform's log|det J| correction. When ``training``
  is true, ``info`` may additionally map ``"aux_losses"`` to a
  ``dict[str, 0-dim Tensor]`` of actor-owned, pre-weighted auxiliary losses.
  The learner adds them to the actor objective (gradients flow through the
  regular actor optimizer and DP gradient sync) and reports each entry as
  ``Loss/aux_<name>``. Auxiliary loss names must be stable across updates.
- ``get_mean_and_std(observations, training) -> (mean, std)``: pre-transform
  distribution parameters.
- ``explore(obs, dones=None, deterministic=False) -> actions``: collection and
  play inference. ``deterministic=True`` must return the transform of the
  distribution mean (the same action ``as_export_module`` produces).
- ``as_export_module() -> nn.Module``: single-input single-output wrapper for
  ONNX export, returning deterministic post-transform actions.
- ``normalize_parameters()``: post-update hook (may be a no-op).
- ``inference_startup_memory_categories(batch_size) -> {"persistent_exploration_scratch": int}``.
- ``action_transform_id``: non-empty ``str`` class/instance attribute naming
  the action transform (e.g. ``"tanh"``); persisted and validated through
  checkpoints.

The same actor instance serves optimization and inference. Custom actors are
incompatible with the whole-cycle CUDA graph: requesting
``use_whole_cycle_cuda_graph`` with ``actor_config`` fails closed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import torch
import torch.nn as nn

ACTOR_SEMANTICS_SCHEMA_VERSION = 1
"""Schema version of the ``actor_semantics`` checkpoint payload."""

BUILTIN_ACTOR_BUILDER = "unilab.rl.algos.flash_sac.network:FlashSACActor"
BUILTIN_ACTION_TRANSFORM_ID = "tanh"

_AUX_LOSS_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+")

_CUSTOM_ACTOR_REQUIRED_ATTRIBUTES = (
    "forward",
    "get_mean_and_std",
    "explore",
    "normalize_parameters",
    "as_export_module",
    "inference_startup_memory_categories",
)


@dataclass(frozen=True)
class FlashSACActorSpec:
    """Spawn-safe declarative description of a custom FlashSAC actor."""

    builder: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, actor_config: Any) -> "FlashSACActorSpec":
        """Parse an owner-config ``actor_config`` mapping, failing closed."""
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(actor_config, (DictConfig, ListConfig)):
            actor_config = OmegaConf.to_container(actor_config, resolve=True)
        if not isinstance(actor_config, Mapping):
            raise TypeError(
                "FlashSAC actor_config must be a mapping with 'builder' and "
                f"optional 'kwargs', got {type(actor_config).__name__}"
            )
        builder = actor_config.get("builder")
        if not isinstance(builder, str) or not builder.strip():
            raise ValueError(
                "FlashSAC actor_config.builder must be a non-empty dotted path "
                f"('pkg.module:callable'), got {builder!r}"
            )
        kwargs = actor_config.get("kwargs") or {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(
                "FlashSAC actor_config.kwargs must be a mapping of "
                f"JSON-serializable values, got {type(kwargs).__name__}"
            )
        kwargs = dict(kwargs)
        try:
            # JSON-serializable kwargs are spawn-safe and checkpointable.
            kwargs = json.loads(json.dumps(kwargs))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"FlashSAC actor_config.kwargs must be JSON-serializable: {exc}"
            ) from exc
        return cls(builder=builder.strip(), kwargs=kwargs)


def _resolve_actor_builder(path: str) -> Any:
    if ":" in path:
        module_path, _, attr_name = path.partition(":")
    else:
        module_path, _, attr_name = path.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(f"Invalid FlashSAC actor builder path: {path!r}")
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import FlashSAC actor builder module {module_path!r}: {exc}. "
            "Ensure the downstream package is installed or importable on "
            "PYTHONPATH in every training process (spawn subprocesses re-import it)."
        ) from exc
    builder = getattr(module, attr_name, None)
    if not callable(builder):
        raise TypeError(f"FlashSAC actor builder {path!r} is not callable")
    return builder


def validate_aux_losses(aux_losses: Any, *, context: str) -> dict[str, torch.Tensor]:
    """Fail closed unless ``aux_losses`` maps stable names to scalar tensors."""
    if aux_losses is None:
        return {}
    if not isinstance(aux_losses, Mapping):
        raise TypeError(
            f"{context}: 'aux_losses' must be a mapping of name -> scalar tensor, "
            f"got {type(aux_losses).__name__}"
        )
    validated: dict[str, torch.Tensor] = {}
    for name, value in aux_losses.items():
        if not isinstance(name, str) or not _AUX_LOSS_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"{context}: auxiliary loss names must match "
                f"{_AUX_LOSS_NAME_PATTERN.pattern!r}, got {name!r}"
            )
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise TypeError(
                f"{context}: auxiliary loss {name!r} must be a 0-dim tensor, "
                f"got {type(value).__name__} with ndim={getattr(value, 'ndim', None)}"
            )
        validated[name] = value
    return validated


def validate_flashsac_actor(actor: nn.Module, *, obs_dim: int, action_dim: int) -> None:
    """Fail closed unless ``actor`` satisfies the custom actor contract.

    Runs a representative forward/explore/export smoke pass and restores model
    and RNG state afterwards, so construction-time validation has no effect on
    training.
    """
    missing = [
        name
        for name in _CUSTOM_ACTOR_REQUIRED_ATTRIBUTES
        if not callable(getattr(actor, name, None))
    ]
    if missing:
        raise TypeError(
            f"Custom FlashSAC actor {type(actor).__name__} is missing required "
            f"contract methods: {', '.join(missing)}. See "
            "unilab.rl.algos.flash_sac.actor_spec for the contract."
        )
    transform_id = getattr(actor, "action_transform_id", None)
    if not isinstance(transform_id, str) or not transform_id:
        raise TypeError(
            f"Custom FlashSAC actor {type(actor).__name__} must declare a "
            "non-empty string 'action_transform_id' identifying its action "
            "transform; it is persisted in checkpoints and validated on load."
        )
    if not any(True for _ in actor.parameters()):
        raise ValueError(
            f"Custom FlashSAC actor {type(actor).__name__} has no parameters; "
            "the learner owns exactly one optimizer over actor.parameters()"
        )

    from unilab.rl.runtime.offpolicy.warmup import capture_rng_state, restore_rng_state

    actor_any = cast(Any, actor)
    device = next(actor.parameters()).device
    saved_state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    rng_state = capture_rng_state(device)
    try:
        with torch.no_grad():
            obs = torch.zeros(2, obs_dim, device=device)
            actions, info = actor(obs, training=True)
            if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (2, action_dim):
                raise TypeError(
                    f"Custom FlashSAC actor forward must return actions with shape "
                    f"(batch, action_dim)={(2, action_dim)}, got "
                    f"{type(actions).__name__} {getattr(actions, 'shape', None)}"
                )
            if not isinstance(info, Mapping) or "log_prob" not in info:
                raise TypeError(
                    "Custom FlashSAC actor forward must return (actions, info) with "
                    "info['log_prob']"
                )
            log_prob = info["log_prob"]
            if not isinstance(log_prob, torch.Tensor) or tuple(log_prob.shape) != (2,):
                raise TypeError(
                    f"Custom FlashSAC actor info['log_prob'] must have shape (batch,)=(2,), "
                    f"got {type(log_prob).__name__} {getattr(log_prob, 'shape', None)}"
                )
            validate_aux_losses(info.get("aux_losses"), context="Custom FlashSAC actor forward")
            for label, sampled in (
                ("deterministic", actor_any.explore(obs, deterministic=True)),
                (
                    "exploration",
                    actor_any.explore(
                        obs, dones=torch.zeros(2, device=device), deterministic=False
                    ),
                ),
            ):
                if not isinstance(sampled, torch.Tensor) or tuple(sampled.shape) != (
                    2,
                    action_dim,
                ):
                    raise TypeError(
                        f"Custom FlashSAC actor explore({label}) must return actions with "
                        f"shape {(2, action_dim)}, got "
                        f"{type(sampled).__name__} {getattr(sampled, 'shape', None)}"
                    )
            export_module = actor_any.as_export_module()
            if not isinstance(export_module, nn.Module):
                raise TypeError("Custom FlashSAC actor as_export_module() must return an nn.Module")
            exported = export_module(obs)
            if not isinstance(exported, torch.Tensor) or tuple(exported.shape) != (
                2,
                action_dim,
            ):
                raise TypeError(
                    f"Custom FlashSAC actor export module must return actions with shape "
                    f"{(2, action_dim)}, got "
                    f"{type(exported).__name__} {getattr(exported, 'shape', None)}"
                )
    finally:
        actor.load_state_dict(saved_state)
        restore_rng_state(rng_state)


def build_custom_actor(
    spec: FlashSACActorSpec,
    *,
    obs_dim: int,
    action_dim: int,
    device: str | torch.device,
) -> nn.Module:
    """Build and validate a custom FlashSAC actor from its declarative spec."""
    builder = _resolve_actor_builder(spec.builder)
    actor = builder(obs_dim=obs_dim, action_dim=action_dim, device=device, **spec.kwargs)
    if not isinstance(actor, nn.Module):
        raise TypeError(
            f"FlashSAC actor builder {spec.builder!r} must return an nn.Module, "
            f"got {type(actor).__name__}"
        )
    validate_flashsac_actor(actor, obs_dim=obs_dim, action_dim=action_dim)
    return actor


def builtin_actor_semantics(
    *,
    obs_dim: int,
    action_dim: int,
    num_blocks: int,
    hidden_dim: int,
    noise_zeta_mu: float,
    noise_zeta_max: int,
) -> dict[str, Any]:
    """Checkpoint semantics payload for the built-in FlashSAC actor."""
    return {
        "schema_version": ACTOR_SEMANTICS_SCHEMA_VERSION,
        "kind": "builtin",
        "builder": BUILTIN_ACTOR_BUILDER,
        "kwargs": {
            "num_blocks": int(num_blocks),
            "hidden_dim": int(hidden_dim),
            "noise_zeta_mu": float(noise_zeta_mu),
            "noise_zeta_max": int(noise_zeta_max),
        },
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_transform": BUILTIN_ACTION_TRANSFORM_ID,
    }


def custom_actor_semantics(
    spec: FlashSACActorSpec,
    *,
    obs_dim: int,
    action_dim: int,
    action_transform_id: str,
) -> dict[str, Any]:
    """Checkpoint semantics payload for a custom actor built from ``spec``."""
    return {
        "schema_version": ACTOR_SEMANTICS_SCHEMA_VERSION,
        "kind": "custom",
        "builder": spec.builder,
        "kwargs": json.loads(json.dumps(spec.kwargs, sort_keys=True)),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_transform": str(action_transform_id),
    }


def validate_actor_semantics(saved: Any, expected: Mapping[str, Any]) -> None:
    """Fail closed when checkpoint actor semantics do not match construction.

    ``saved`` is the ``actor_semantics`` payload stored in the checkpoint (or
    ``None`` for checkpoints written before this contract existed). Legacy
    payloads only load into the built-in actor; any configured ``actor_config``
    requires a matching recorded payload.
    """
    if saved is None:
        if expected.get("kind") != "builtin":
            raise ValueError(
                "Checkpoint does not record actor semantics (pre-contract format); "
                "it can only be loaded with the built-in FlashSAC actor. Remove "
                "algo.algo_params.actor_config or retrain with the custom actor."
            )
        return
    if not isinstance(saved, Mapping):
        raise TypeError(
            f"Checkpoint 'actor_semantics' must be a mapping, got {type(saved).__name__}"
        )
    if saved.get("schema_version") != ACTOR_SEMANTICS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint actor semantics schema version: "
            f"{saved.get('schema_version')!r} (expected {ACTOR_SEMANTICS_SCHEMA_VERSION})"
        )
    mismatches = []
    for key in ("kind", "builder", "kwargs", "obs_dim", "action_dim", "action_transform"):
        saved_value = saved.get(key)
        expected_value = expected.get(key)
        if key == "kwargs":
            saved_value = json.loads(json.dumps(saved_value or {}, sort_keys=True))
            expected_value = json.loads(json.dumps(expected_value or {}, sort_keys=True))
        if saved_value != expected_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r} != config={expected_value!r}")
    if mismatches:
        raise ValueError(
            "Checkpoint actor semantics do not match the configured actor; "
            "refusing to load incompatible policy weights: " + "; ".join(mismatches)
        )
