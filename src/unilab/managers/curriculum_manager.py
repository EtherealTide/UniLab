# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/curriculum_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for the tensor-only Manager runtime; licensed under Apache-2.0.
"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term.

    Curriculum terms modify environment parameters during training to implement
    curriculum learning strategies (e.g., gradually increasing task difficulty).
    """

    pass


class CurriculumManager(ManagerBase):
    """Manages curriculum learning for the environment.

    The curriculum manager updates environment parameters during training based
    on agent performance. Each term can modify different aspects of the task
    difficulty (e.g., terrain complexity, command ranges).
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, CurriculumTermCfg | None], env: ManagerBasedRlEnv):
        self._term_names: list[str] = list()
        self._term_cfgs: list[CurriculumTermCfg] = list()
        self._class_term_cfgs: list[CurriculumTermCfg] = list()

        self.cfg = deepcopy(cfg)
        super().__init__(env)

        self._curriculum_state: dict[str, Any] = {}
        for term_name in self._term_names:
            self._curriculum_state[term_name] = None

    def __str__(self) -> str:
        msg = f"<CurriculumManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Curriculum Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        msg += str(table.get_string())
        msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    # Methods.

    def get_term_cfg(self, term_name: str) -> CurriculumTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                data = []
                if isinstance(term_state, dict):
                    for _key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().cpu().item()
                        data.append(value)
                else:
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.detach().cpu().item()
                    data.append(term_state)
                terms.append((term_name, data))
        return terms

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        """Publish curriculum state and reset mask-aware term state."""
        extras: dict[str, float] = {}
        for term_name, term_state in self._curriculum_state.items():
            if term_state is None:
                continue
            if isinstance(term_state, dict):
                for key, value in term_state.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    extras[f"Curriculum/{term_name}/{key}"] = value
            else:
                if isinstance(term_state, torch.Tensor):
                    term_state = term_state.detach().cpu().item()
                extras[f"Curriculum/{term_name}"] = term_state
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(reset_mask)
        return extras

    def compute(self, reset_mask: torch.Tensor | None = None) -> None:
        """Compute global and, when declared, mask-scoped curriculum terms."""
        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
            callable_mask = getattr(term_cfg.func, "__call_reset__", None)
            if callable(callable_mask):
                if reset_mask is None:
                    raise ValueError(f"CurriculumManager term '{name}' requires a reset mask")
                state = callable_mask(self._env, reset_mask, **term_cfg.params)
            else:
                state = term_cfg.func(self._env, slice(None), **term_cfg.params)
            self._validate_state(name, state)
            self._curriculum_state[name] = state

    def compute_reset(self, reset_mask: torch.Tensor) -> None:
        """Compute only terms that own a mask-scoped reset boundary."""
        self.compute(reset_mask)

    def _validate_state(self, term_name: str, state: Any) -> None:
        values = state.values() if isinstance(state, dict) else (state,)
        for value in values:
            if isinstance(value, torch.Tensor):
                finite = bool(torch.isfinite(value).all())
            elif isinstance(value, (int, float)):
                finite = bool(math.isfinite(value))
            else:
                continue
            if not finite:
                raise ValueError(f"CurriculumManager term '{term_name}' returned NaN or Inf.")

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)


class NullCurriculumManager:
    """Placeholder for absent curriculum manager that safely no-ops all operations."""

    def __init__(self):
        self.active_terms: list[str] = []
        self._curriculum_state: dict[str, Any] = {}
        self.cfg = None

    def __str__(self) -> str:
        return "<NullCurriculumManager> (inactive)"

    def __repr__(self) -> str:
        return "NullCurriculumManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        del reset_mask
        return {}

    def compute(self, reset_mask: torch.Tensor | None = None) -> None:
        del reset_mask
