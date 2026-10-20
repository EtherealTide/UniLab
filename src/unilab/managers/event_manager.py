# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/event_manager.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from prettytable import PrettyTable

from unilab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from unilab.managers.rng import ManagerRng

if TYPE_CHECKING:
    from unilab.managers.types import ManagerBasedRlEnv

EventMode = Literal["startup", "reset", "interval", "step"]


@dataclass(kw_only=True)
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for an event term.

    Event terms trigger operations at specific simulation events. They're commonly
    used for domain randomization, state resets, and periodic perturbations.

    The four modes determine when the event fires:

    - ``"startup"``: Once when the environment initializes. Use for parameters that
      should be randomized per-environment but stay constant within an episode (e.g.,
      domain randomization).

    - ``"reset"``: On every episode reset. Use for parameters that should vary between
      episodes (e.g., initial robot pose, domain randomization).

    - ``"interval"``: Periodically during simulation, controlled by ``interval_range_s``.
      Use for perturbations that should happen during episodes (e.g., pushing the robot,
      external disturbances).

    - ``"step"``: Every environment step, unconditionally on all envs. Use for terms that
      manage per-step state such as force lifetimes (e.g., ``apply_body_impulse``).
    """

    mode: EventMode
    """When the event triggers: ``"startup"`` (once at init), ``"reset"`` (every
  episode), ``"interval"`` (periodically during simulation), or ``"step"`` (every
  environment step)."""

    interval_range_s: tuple[float, float] | None = None
    """Time range in seconds for interval mode. The next trigger time is uniformly
  sampled from ``[min, max]``. Required when ``mode="interval"``."""

    is_global_time: bool = False
    """Whether all environments share the same timer. If True, all envs trigger
  simultaneously. If False (default), each env has an independent timer that
  resets on episode reset. Only applies to ``mode="interval"``."""

    min_step_count_between_reset: int = 0
    """Minimum environment steps between triggers. Prevents the event from firing
  too frequently when episodes reset rapidly. Only applies to ``mode="reset"``.
  Set to 0 (default) to trigger on every reset."""


class EventManager(ManagerBase):
    """Manages event-based operations for the environment.

    The event manager triggers operations at different simulation events: startup
    (once at initialization), reset (on episode reset), or interval (periodically
    during simulation). Common uses include domain randomization and state resets.
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, EventTermCfg | None], env: ManagerBasedRlEnv):
        self.cfg = deepcopy(cfg)
        self._device = torch.device(getattr(env, "device", torch.device("cpu")))
        rng = env.rng
        if not isinstance(rng, ManagerRng):
            raise RuntimeError(
                "EventManager interval sampling requires the Manager-owned Torch generator"
            )
        self._rng = rng
        self._mode_term_names: dict[EventMode, list[str]] = dict()
        self._mode_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[EventMode, list[EventTermCfg]] = dict()

        super().__init__(env=env)

    def __str__(self) -> str:
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
        for mode in self._mode_term_names:
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(
                    zip(self._mode_term_names[mode], self._mode_term_cfgs[mode], strict=False)
                ):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
                    table.add_row([index, name])
            msg += str(table.get_string())
            msg += "\n"
        return msg

    # Properties.

    @property
    def active_terms(self) -> dict[EventMode, list[str]]:
        return self._mode_term_names

    @property
    def term_cfgs(self) -> dict[str, EventTermCfg]:
        """Return active event term configs keyed by their global term name."""
        return {
            name: self._mode_term_cfgs[mode][index]
            for mode, names in self._mode_term_names.items()
            for index, name in enumerate(names)
        }

    @property
    def available_modes(self) -> list[EventMode]:
        return list(self._mode_term_names.keys())

    @property
    def uses_startup_rows(self) -> bool:
        """Whether any startup term requires concrete full-width row indices."""
        return any(
            bool(getattr(term_cfg.func, "uses_startup_rows", False))
            for term_cfg in self._mode_term_cfgs.get("startup", ())
        )

    # Methods.

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Get the configuration of a specific event term by name."""
        for mode in self._mode_term_names:
            if term_name in self._mode_term_names[mode]:
                index = self._mode_term_names[mode].index(term_name)
                return self._mode_term_cfgs[mode][index]
        raise ValueError(f"Event term '{term_name}' not found in active terms.")

    def reset(self, reset_mask: torch.Tensor) -> dict[str, float]:
        """Clear stateful event terms through the fixed-width mask."""
        self._validate_reset_mask(reset_mask)
        for mode, mode_cfg in self._mode_class_term_cfgs.items():
            if mode == "reset":
                continue
            for term_cfg in mode_cfg:
                term_cfg.func.reset(reset_mask)
        # Iterate the full interval term list: _interval_term_time_left is parallel
        # to _mode_term_cfgs["interval"], not the class-only subset.
        for index, term_cfg in enumerate(self._mode_term_cfgs.get("interval", ())):
            if not term_cfg.is_global_time:
                assert term_cfg.interval_range_s is not None
                lower, upper = term_cfg.interval_range_s
                sampled = self._sample_interval(lower, upper, self.num_envs)
                timer = self._interval_term_time_left[index]
                timer.copy_(torch.where(reset_mask, sampled, timer))
        return {}

    def apply(
        self,
        mode: EventMode,
        env_ids: torch.Tensor | slice | None = None,
        dt: float | None = None,
    ):
        if mode not in ("startup", "reset", "interval", "step"):
            raise ValueError(f"Unsupported event mode '{mode}'.")
        if mode not in self._mode_term_cfgs:
            return
        if mode == "interval" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
        if mode == "interval" and env_ids is not None:
            raise ValueError(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )
        if mode == "reset":
            raise ValueError("Event reset terms are dispatched through dispatch_reset(mask).")
        if mode == "step" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")

        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            if mode == "interval":
                time_left = self._interval_term_time_left[index]
                assert dt is not None
                time_left -= dt
                if term_cfg.is_global_time:
                    if time_left < 1e-6:
                        assert term_cfg.interval_range_s is not None
                        lower, upper = term_cfg.interval_range_s
                        self._interval_term_time_left[index][:] = self._sample_interval(
                            lower, upper, 1
                        )
                        term_cfg.func(self._env, None, **term_cfg.params)
                else:
                    valid_env_ids = torch.nonzero(time_left < 1e-6, as_tuple=False).flatten()
                    if valid_env_ids.numel() > 0:
                        assert term_cfg.interval_range_s is not None
                        lower, upper = term_cfg.interval_range_s
                        self._interval_term_time_left[index][valid_env_ids] = self._sample_interval(
                            lower, upper, valid_env_ids.numel()
                        )
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            elif mode == "step":
                term_cfg.func(self._env, None, **term_cfg.params)
            else:
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    def dispatch_reset(
        self,
        reset_mask: torch.Tensor,
        *,
        global_env_step_count: int,
    ) -> None:
        """Dispatch fixed-shape reset events."""
        self._validate_reset_mask(reset_mask)
        for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            term_func = getattr(term_cfg, "func", None)
            if not callable(getattr(term_func, "reset", None)):
                raise NotImplementedError(
                    f"EventManager term '{self._mode_term_names['reset'][index]}' does not "
                    "define the mask-preserving reset boundary; func="
                    f"{getattr(term_func, '__name__', type(term_func).__name__)}"
                )
            self._reset_term_last_triggered_step_id[index].masked_fill_(
                reset_mask, global_env_step_count
            )
            self._reset_term_last_triggered_once[index].masked_fill_(reset_mask, True)
            assert term_func is not None
            term_func.reset(self._env, reset_mask, **term_cfg.params)

    def _validate_reset_mask(self, reset_mask: torch.Tensor) -> None:
        if reset_mask.ndim != 1 or reset_mask.shape[0] != self.num_envs:
            raise ValueError(f"EventManager reset mask must have shape {(self.num_envs,)}")
        if reset_mask.dtype != torch.bool or reset_mask.device != self._device:
            raise ValueError(f"EventManager reset mask must be bool on {self._device}")

    def _prepare_terms(self) -> None:
        self._interval_term_time_left: list[torch.Tensor] = list()
        self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
        self._reset_term_last_triggered_once: list[torch.Tensor] = list()

        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            if term_cfg.mode not in ("startup", "reset", "interval", "step"):
                raise ValueError(
                    f"EventManager term '{term_name}' has unsupported mode '{term_cfg.mode}'."
                )
            if term_cfg.mode not in self._mode_term_names:
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)
            if (
                term_cfg.mode in ("step", "interval")
                and hasattr(term_cfg.func, "reset")
                and callable(term_cfg.func.reset)
            ):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)
            if term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset:
                    raise NotImplementedError(
                        f"EventManager reset term '{term_name}' uses "
                        "min_step_count_between_reset, which is incompatible with the "
                        "fixed-width mask reset contract"
                    )
                if not callable(getattr(term_cfg.func, "reset", None)):
                    raise NotImplementedError(
                        f"EventManager reset term '{term_name}' does not define the "
                        "mask-preserving reset boundary"
                    )
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                lower, upper = term_cfg.interval_range_s
                if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
                    raise ValueError(
                        f"EventManager term '{term_name}' has invalid interval_range_s "
                        f"{term_cfg.interval_range_s}."
                    )
                shape = 1 if term_cfg.is_global_time else self.num_envs
                self._interval_term_time_left.append(self._sample_interval(lower, upper, shape))
            elif term_cfg.mode == "reset":
                step_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
                self._reset_term_last_triggered_step_id.append(step_count)
                no_trigger = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
                self._reset_term_last_triggered_once.append(no_trigger)

            func = term_cfg.func
            if hasattr(func, "model_fields"):
                raise NotImplementedError(
                    f"EventManager term '{term_name}' requests direct model-field mutation; "
                    "this capability is unsupported by the standalone UniLab manager core."
                )

    def _sample_interval(self, lower: float, upper: float, count: int) -> torch.Tensor:
        return self._rng.uniform(lower, upper, (count,), dtype=torch.float64).reshape(count)
