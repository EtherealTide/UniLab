"""Cold semantic-owner contract for the FlashSAC G1 motion tasks.

The training runtime is the sole Manager-Based environment.  This module keeps
only the cold owner fingerprint and fail-closed validation; every hot lifecycle
component is owned by Manager terms and the public UniSim tensor boundaries.
"""

from __future__ import annotations

from typing import Any

from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.common.tensor_runtime import semantic_fingerprint


def _qualified_name(value: Any) -> str:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _torch_g1_flashsac_owner_identity(cfg: ManagerBasedRlEnvCfg) -> str:
    """Return a versioned semantic fingerprint for the narrow FlashSAC owner."""
    if cfg.fixed_model_variants is not None:
        raise ValueError("Torch G1 FlashSAC v1 does not support fixed model variants")
    if cfg.scene is None:
        raise ValueError("Torch G1 FlashSAC requires a scene owner")
    return semantic_fingerprint(
        "unilab.motion_tracking.g1.tensor.v1",
        (
            1,
            cfg.scene,
            cfg.sim_dt,
            cfg.ctrl_dt,
            cfg.max_episode_seconds,
            cfg.observations,
            cfg.actions,
            cfg.commands,
            cfg.rewards,
            cfg.terminations,
            cfg.events,
            cfg.curriculum,
            cfg.metrics,
            cfg.recorders,
            cfg.auto_reset,
            cfg.is_finite_horizon,
            cfg.scale_rewards_by_dt,
            cfg.policy_observation_group,
            cfg.critic_observation_group,
        ),
    )


_TORCH_G1_MANAGER_TERMS_OWNER_IDENTITY_V1 = (
    "2759440606297f25313fa662b7a34142db93f9b6418c07728aac9e5f35bb6d5c"
)


def _validate_torch_g1_flashsac_owner_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    identity = _torch_g1_flashsac_owner_identity(cfg)
    if identity != _TORCH_G1_MANAGER_TERMS_OWNER_IDENTITY_V1:
        raise ValueError(
            "Torch G1 tensor runtime supports only canonical owner contracts; "
            f"semantic identity {identity} is not recognized"
        )


__all__ = [
    "_TORCH_G1_MANAGER_TERMS_OWNER_IDENTITY_V1",
    "_qualified_name",
    "_torch_g1_flashsac_owner_identity",
    "_validate_torch_g1_flashsac_owner_contract",
]
