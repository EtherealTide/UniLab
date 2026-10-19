"""Shared training seed contract helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class TrainingSeedInfo:
    """Configured and effective seed metadata for a training run."""

    configured_seed: int | None
    configured_seed_source: str | None
    effective_seed: int | None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "configured_seed": self.configured_seed,
            "configured_seed_source": self.configured_seed_source,
            "effective_seed": self.effective_seed,
        }


def _select_seed(cfg: Any, path: str) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=None)

    current = cfg
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def resolve_training_seed(cfg: Any) -> TrainingSeedInfo:
    """Resolve the configured seed from the algorithm-level ``algo.seed`` contract."""
    raw_seed = _select_seed(cfg, "algo.seed")
    if raw_seed is None:
        if _select_seed(cfg, "training.seed") is not None:
            raise ValueError("training.seed is no longer supported; configure algo.seed instead")
        return TrainingSeedInfo(
            configured_seed=None, configured_seed_source=None, effective_seed=None
        )
    seed = int(raw_seed)
    if seed < 0:
        raise ValueError(f"algo.seed must be non-negative, got {seed}")
    return TrainingSeedInfo(
        configured_seed=seed,
        configured_seed_source="algo.seed",
        effective_seed=seed,
    )


def apply_training_seed(
    seed: int | None,
    *,
    torch_runtime: bool = True,
    cuda: bool = True,
) -> int | None:
    """Apply a seed to the runtimes used by training entrypoints."""
    if seed is None:
        return None

    effective_seed = int(seed)
    if effective_seed < 0:
        raise ValueError(f"seed must be non-negative, got {effective_seed}")

    random.seed(effective_seed)

    if torch_runtime:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.manual_seed(effective_seed)
            if cuda and torch.cuda.is_available():
                torch.cuda.manual_seed_all(effective_seed)

    return effective_seed


def apply_configured_training_seed(
    cfg: Any,
    *,
    torch_runtime: bool = True,
    cuda: bool = True,
) -> TrainingSeedInfo:
    """Resolve and apply the configured training seed before runtime construction."""
    seed_info = resolve_training_seed(cfg)
    effective_seed = apply_training_seed(
        seed_info.effective_seed,
        torch_runtime=torch_runtime,
        cuda=cuda,
    )
    return TrainingSeedInfo(
        configured_seed=seed_info.configured_seed,
        configured_seed_source=seed_info.configured_seed_source,
        effective_seed=effective_seed,
    )
