# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/metrics_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Metrics manager for logging custom per-step metrics during training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv

REDUCE_OPTIONS = ("last", "max", "mean", "sum")


@dataclass(kw_only=True)
class MetricsTermCfg(ManagerTermBaseCfg):
    """Configuration for a metrics term.

    Attributes:
      per_substep: If True, evaluate this term once per physics substep inside
        the decimation loop and report the per-step mean. Only the integrated
        state (qpos, qvel, act) is current mid-loop; all derived quantities
        (xpos, xquat, site_xpos, actuator_force, contacts, ...) are stale.

      reduce: How to aggregate per-step values into an episode metric.
        - ``"mean"`` (default) reports ``sum / step_count``.
        - ``"last"`` reports the value from the final step of the episode,
          useful for binary success metrics that should not be averaged over
          timesteps.
        - ``"max"`` reports the highest value seen during the episode, useful
          for peak metrics like maximum power or contact force.
        - ``"sum"`` reports the accumulated total over the episode, useful for
          cumulative quantities like episodic reward or total distance
          traveled.
    """

    per_substep: bool = False
    reduce: Literal["last", "max", "mean", "sum"] = "mean"


class MetricsManager(ManagerBase):
    """Accumulates per-step metric values, reports episode averages.

    Unlike rewards, metrics have no weight, no dt scaling, and no
    normalization by episode length. Episode values are true per-step
    averages (sum / step_count), so a metric in [0,1] stays in [0,1]
    in the logger.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, MetricsTermCfg | None], env: ManagerBasedRlEnv):
        self._term_names: list[str] = list()
        self._term_cfgs: list[MetricsTermCfg] = list()
        self._class_term_cfgs: list[MetricsTermCfg] = list()
        self._step_term_indices: list[int] = list()
        self._substep_term_indices: list[int] = list()

        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._device = getattr(env, "device", torch.device("cpu"))
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._episode_max: dict[str, torch.Tensor] = {}
        for idx, term_name in enumerate(self._term_names):
            if self._term_cfgs[idx].reduce not in REDUCE_OPTIONS:
                msg = (
                    f"The reduce method '{self._term_cfgs[idx].reduce}' for metric '{term_name}' "
                    f"is unknown. Valid options are {REDUCE_OPTIONS}."
                )
                raise ValueError(msg)

            self._episode_sums[term_name] = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self._device
            )

            if self._term_cfgs[idx].reduce == "max":
                self._episode_max[term_name] = torch.full(
                    (self.num_envs,), float("-inf"), dtype=torch.float32, device=self._device
                )
        # Pre-resolved tensor refs for substep terms to avoid dict lookups in
        # the hot loop.
        self._substep_accum: list[torch.Tensor] = []
        self._substep_episode_sums: list[torch.Tensor] = []
        self._substep_episode_max: list[torch.Tensor | None] = []
        for idx in self._substep_term_indices:
            name = self._term_names[idx]
            buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self._device)
            self._substep_accum.append(buf)
            self._substep_episode_sums.append(self._episode_sums[name])
            self._substep_episode_max.append(self._episode_max.get(name))
        self._substep_count: int = 0
        self._step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self._step_values = torch.zeros(
            (self.num_envs, len(self._term_names)), dtype=torch.float32, device=self._device
        )

    def __str__(self) -> str:
        msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Metrics Terms"
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

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        mask = self._reset_mask(env_ids)
        counts = self._step_count[mask].to(torch.float32)
        # Avoid division by zero for envs that haven't stepped.
        safe_counts = torch.clamp(counts, min=1.0)
        for idx, key in enumerate(self._episode_sums):
            reduce = self._term_cfgs[idx].reduce
            if reduce == "max":
                values = self._episode_max[key][mask]
                extras["Episode_Metrics/" + key] = self._log_mean(values)

            elif reduce == "last":
                extras["Episode_Metrics/" + key] = self._log_mean(self._step_values[mask, idx])

            elif reduce == "sum":
                extras["Episode_Metrics/" + key] = self._log_mean(self._episode_sums[key][mask])

            else:
                values = self._episode_sums[key][mask] / safe_counts
                extras["Episode_Metrics/" + key] = self._log_mean(values)

        self.clear_episode_state(env_ids)
        return extras

    def clear_episode_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear selected episode accumulators without publishing summaries.

        Reset owners consume this boundary when they own metric reset. Generic
        reset behavior remains publication followed by this same selected-row
        state clear.
        """
        if env_ids is None:
            env_ids = slice(None)
        mask = self._reset_mask(env_ids)
        for key in self._episode_sums:
            self._episode_sums[key].masked_fill_(mask, 0.0)
            if key in self._episode_max:
                self._episode_max[key].masked_fill_(mask, float("-inf"))
        self._step_count.masked_fill_(mask, 0)
        for buf in self._substep_accum:
            buf.masked_fill_(mask, 0.0)
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

    def compute_substep(self) -> None:
        """Accumulate per-substep metric values inside the decimation loop.

        No-op when no ``per_substep`` terms are configured.
        """
        if not self._substep_term_indices:
            return
        for i, idx in enumerate(self._substep_term_indices):
            value = self._compute_term(idx)
            self._substep_accum[i] += value
        self._substep_count += 1

    def compute(self) -> None:
        self._step_count += 1
        if self._substep_term_indices and self._substep_count > 0:
            for i, idx in enumerate(self._substep_term_indices):
                avg = self._substep_accum[i] / self._substep_count
                self._substep_episode_sums[i] += avg
                self._step_values[:, idx] = avg
                max_buf = self._substep_episode_max[i]
                if max_buf is not None:
                    torch.maximum(max_buf, avg, out=max_buf)
                    self._substep_accum[i].fill_(0.0)
            self._substep_count = 0
        for idx in self._step_term_indices:
            name = self._term_names[idx]
            value = self._compute_term(idx)
            self._episode_sums[name] += value
            self._step_values[:, idx] = value
            if name in self._episode_max:
                torch.maximum(self._episode_max[name], value, out=self._episode_max[name])

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_values[env_idx, idx].item()]))
        return terms

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            idx = len(self._term_names)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if term_cfg.per_substep:
                self._substep_term_indices.append(idx)
            else:
                self._step_term_indices.append(idx)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)

    def _compute_term(self, idx: int) -> torch.Tensor:
        name = self._term_names[idx]
        term_cfg = self._term_cfgs[idx]
        value = term_cfg.func(self._env, **term_cfg.params)
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float32:
                raise TypeError(
                    f"MetricsManager term '{name}' returned dtype {value.dtype}, expected float32."
                )
            if value.device != self._device:
                raise ValueError(
                    f"MetricsManager term '{name}' returned device {value.device}, "
                    f"expected {self._device}."
                )
            result = value.clone()
        else:
            host = np.array(value, dtype=np.float32, order="C", copy=True)
            result = torch.from_numpy(host).to(device=self._device)
        if result.shape != (self.num_envs,):
            raise ValueError(
                f"MetricsManager term '{name}' returned shape {tuple(result.shape)}; "
                f"expected ({self.num_envs},)."
            )
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"MetricsManager term '{name}' returned non-finite values.")
        return result

    def _reset_mask(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
        if env_ids is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self._device)
        if isinstance(env_ids, slice):
            indices = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)[env_ids]
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
            return mask.index_fill(0, indices, True)
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.ndim != 1
            or env_ids.dtype
            not in {
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("MetricsManager reset rows must be one-dimensional integers")
        rows = env_ids.to(self._device)
        if rows.numel() and (rows.min() < 0 or rows.max() >= self.num_envs):
            raise IndexError(f"MetricsManager reset rows out of range: {rows.tolist()}")
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        return mask.index_fill(0, rows.to(torch.int64), True)

    @staticmethod
    def _log_mean(values: torch.Tensor) -> float:
        if values.numel() == 0:
            return 0.0
        return float(values.mean().item())


class NullMetricsManager:
    """Placeholder for absent metrics manager that safely no-ops all operations."""

    def __init__(self):
        self.active_terms: list[str] = []
        self.cfg = None

    def __str__(self) -> str:
        return "<NullMetricsManager> (inactive)"

    def __repr__(self) -> str:
        return "NullMetricsManager()"

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        return {}

    def compute_substep(self) -> None:
        pass

    def compute(self) -> None:
        pass
