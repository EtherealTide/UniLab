"""Manager-owned reset lifecycle declarations for fused task owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResetOwnerCfg(ManagerTermBaseCfg):
    """Declare which ordinary reset passes a fused owner replaces.

    A reset owner remains a Manager-owned component: it is declared by Manager
    config, executes inside the Manager reset lifecycle, consumes public
    contracts, and never becomes a second environment entry point. Ownership is
    explicit so semantic fingerprints and call-graph tests can prove which
    generic passes are replaced rather than inferring it from implementation
    details.
    """

    command_name: str = "motion"
    owns_command_reset: bool = True
    owns_action_reset: bool = True
    owns_observation_reset: bool = True
    owns_metric_reset: bool = True


class ResetOwner(ManagerTermBase):
    """Fail-closed base class for fused reset lifecycle owners."""

    cfg: ResetOwnerCfg

    def __init__(self, cfg: ResetOwnerCfg, env: ManagerBasedRlEnv):
        self.cfg = cfg
        super().__init__(env)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        """Execute the declared selected-reset ownership in Manager lifecycle."""
        raise NotImplementedError(type(self).__name__)


class ResetOwnerManager:
    """Construct and expose the sole configured reset owner.

    Multiple reset owners would reintroduce reset-pass ordering ambiguity. The
    Manager therefore accepts at most one owner and fails closed otherwise.
    """

    def __init__(self, cfg: dict[str, ResetOwnerCfg | None], env: ManagerBasedRlEnv):
        self.cfg = dict(cfg)
        self._terms: dict[str, ResetOwner] = {}
        active = [(name, term_cfg) for name, term_cfg in self.cfg.items() if term_cfg is not None]
        if len(active) > 1:
            names = ", ".join(name for name, _ in active)
            raise ValueError(f"ManagerBased reset supports at most one reset owner; got {names}")
        for name, term_cfg in active:
            term = term_cfg.func(cfg=term_cfg, env=env)  # type: ignore[call-arg]
            if not isinstance(term, ResetOwner):
                raise TypeError(
                    f"Reset owner '{name}' built {type(term).__name__}, expected ResetOwner"
                )
            self._terms[name] = term

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms)

    @property
    def owner(self) -> ResetOwner | None:
        if not self._terms:
            return None
        return next(iter(self._terms.values()))

    def get_term(self, name: str) -> ResetOwner:
        try:
            return self._terms[name]
        except KeyError as exc:
            raise KeyError(f"Reset owner '{name}' is not configured") from exc

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        for term in self._terms.values():
            term.reset(env_ids)

    def __str__(self) -> str:
        if not self._terms:
            return "<ResetOwnerManager> (inactive)"
        names = ", ".join(self._terms)
        return f"<ResetOwnerManager> {names}"


class NullResetOwnerManager:
    """No-op peer for owners without a reset-owner declaration."""

    def __init__(self) -> None:
        self.active_terms: list[str] = []
        self.cfg = None

    @property
    def owner(self) -> None:
        return None

    def get_term(self, name: str) -> None:
        raise KeyError(f"Reset owner '{name}' is not configured")

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        del env_ids

    def __str__(self) -> str:
        return "<ResetOwnerManager> (inactive)"


__all__ = [
    "NullResetOwnerManager",
    "ResetOwner",
    "ResetOwnerCfg",
    "ResetOwnerManager",
]
