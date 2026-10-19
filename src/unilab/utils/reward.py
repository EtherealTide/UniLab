"""Utility functions for reward config handling."""

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

RewardDict = dict[str, Any]


def _to_reward_dict(value: object, *, error_message: str) -> RewardDict:
    """Convert an OmegaConf container into a plain reward dictionary."""
    resolved = OmegaConf.to_container(value, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(error_message)
    # Some reward configs are mounted as a full `reward:` section.
    # Env config injection expects the inner reward mapping.
    if set(resolved) == {"reward"} and isinstance(resolved["reward"], dict):
        return cast(RewardDict, resolved["reward"])
    return cast(RewardDict, resolved)


def resolve_reward_dict(
    cfg: DictConfig, *, target_field: str = "reward_config"
) -> dict[str, RewardDict]:
    """Extract and validate the reward config from a composed Hydra config.

    Args:
        cfg: Hydra DictConfig containing the root ``reward`` section.
        target_field: Env config override key the mapping is mounted under.

    Returns:
        Dictionary with ``target_field`` as the env config override key.

    Raises:
        ValueError: If the reward config is missing or empty, or the
            ``target_field`` is not a supported override key.
    """
    if target_field not in {"reward_config", "rewards"}:
        raise ValueError(
            f"Reward config target_field must be 'reward_config' or 'rewards', got {target_field!r}"
        )
    reward_cfg = OmegaConf.select(cfg, "reward")
    if not reward_cfg:
        raise ValueError(
            "Missing 'reward' config in Hydra. Reward config must be explicitly provided."
        )

    reward_dict = _to_reward_dict(
        reward_cfg,
        error_message="Reward config must resolve to a mapping.",
    )
    if not reward_dict:
        raise ValueError(
            "Reward config resolved to empty. Please select a non-default reward override."
        )

    return {target_field: reward_dict}
