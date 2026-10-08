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
            cfg.reset_owners,
            cfg.auto_reset,
            cfg.is_finite_horizon,
            cfg.scale_rewards_by_dt,
            cfg.policy_observation_group,
            cfg.critic_observation_group,
        ),
    )


_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V10 = (
    "ea611d71bdc07c58c022e9048b2d91d59587dc71d2de442aba881fef02e36e84"
)
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11 = (
    "2bc1ee4bd3c6a9c889c0f6e46c566b183900f5d20965f27d6137369f122feb95"
)
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V12 = (
    "2bc1ee4bd3c6a9c889c0f6e46c566b183900f5d20965f27d6137369f122feb95"
)
# V13 applies only to MJWarp: the privileged critic carrier is fused into one
# Manager-owned observation term. Ordering, equations, dimensions, actor noise,
# and the clean critic corruption policy are unchanged.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V13 = (
    "73fa809d1acaddab43fbf3b99d67e4b0f9464ddf8ab6ff33fe6cb1e23e8c043b"
)
# V14 applies only to MJWarp: the canonical action-rate, joint-limit, and
# undesired-contact penalties are fused into one Manager-owned reward term.
# Equations, weights, per-term log keys, and reward ordering are unchanged.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V14 = (
    "b622c4da96c4e28595ed4900a7bf539c18437d4917c7c529dd3c2850a77f3061"
)
# V15 applies only to MJWarp: the canonical anchor-position, anchor-orientation,
# and end-effector-position failure terms are fused into one Manager-owned
# termination term. Equations, thresholds, and timeout semantics are unchanged.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V15 = (
    "29d3a052d0bb1cdab7eb5811477c32e16a3275236815fca8d6fa109e58b07bca"
)
# V16 applies only to MJWarp: the two fused reward owners may skip the Manager's
# defensive result copy by declaring their stable per-call output transient, and
# the sampler keeps its adaptive alpha scalar on-device. Equations, weights,
# per-term logs, RNG, and all public reward values are unchanged.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V16 = (
    "bdd459e6cd54bfb3535a825f762b3ab57d7ed47b0bc16fd7a7b6cc4e9a1cc9c1"
)
# V17-V20 were transition identities while packed logging moved from internal
# pack weights to canonical RewardManager term weights. They remain accepted so
# in-flight owner snapshots fail closed only on semantic drift, not on this
# completed transition.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V17 = (
    "205268ac43ca0b1b9eaaf5ec3d62a1e28b7ff7051d3ea5fcb7d56fee03c9fb19"
)
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V18 = (
    "e4ed72fe9d32e3eaafc4cddfbf70d43d39ba514b0175b5aa4b74b0727fdf515d"
)
# V21 applies to the non-MJWarp owners and V22 to MJWarp: the fused packs emit
# raw per-component columns, while RewardManager owns canonical names/weights.
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V21 = (
    "8d554342b93da489faaa1cd3f8a1be5ed242960f3c26d890230d1dee80fd654c"
)
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V22 = (
    "5f406476f3b2cf9e8cafbcbe2b4f67562f5d223f421f1324ccbf3f0aee600d60"
)
_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V23 = (
    "62c8f187d88a7c1a85b71cb88f153b61aaf350acecea80e45c8109f5f532ddbc"
)
_TORCH_G1_SAC_OWNER_IDENTITY_V2 = "a4cd74dd23a511de02c572102856190d78b819f874ec7d81e104aaff03e82c80"


def _validate_torch_g1_flashsac_owner_contract(cfg: ManagerBasedRlEnvCfg) -> None:
    identity = _torch_g1_flashsac_owner_identity(cfg)
    if identity not in {
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V10,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V12,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V13,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V14,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V15,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V16,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V17,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V18,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V21,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V22,
        _TORCH_G1_FLASHSAC_OWNER_IDENTITY_V23,
        _TORCH_G1_SAC_OWNER_IDENTITY_V2,
    }:
        raise ValueError(
            "Torch G1 tensor runtime supports only canonical owner contracts; "
            f"semantic identity {identity} is not recognized"
        )


__all__ = [
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V10",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V12",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V13",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V14",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V15",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V16",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V17",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V18",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V21",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V22",
    "_TORCH_G1_FLASHSAC_OWNER_IDENTITY_V23",
    "_TORCH_G1_SAC_OWNER_IDENTITY_V2",
    "_qualified_name",
    "_torch_g1_flashsac_owner_identity",
    "_validate_torch_g1_flashsac_owner_contract",
]
