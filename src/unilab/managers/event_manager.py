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
from unilab.managers.torch_rng import TorchManagerRng

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv

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
        rng = env.torch_rng
        if not isinstance(rng, TorchManagerRng):
            raise RuntimeError(
                "EventManager interval sampling requires the Manager-owned Torch generator"
            )
        self._torch_rng = rng
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
    def uses_tensor_reset_rows(self) -> bool:
        """Whether every reset term can consume a device row selector."""
        return bool(self._mode_term_cfgs.get("reset")) and all(
            bool(getattr(term_cfg.func, "uses_tensor_rows", False))
            for term_cfg in self._mode_term_cfgs["reset"]
        )

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

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        ids: torch.Tensor | slice = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.arange(self.num_envs, device=self._device)[
                env_ids if env_ids is not None else slice(None)
            ]
        )
        num_envs = ids.numel() if isinstance(ids, torch.Tensor) else self.num_envs
        # Iterate the full interval term list: _interval_term_time_left is parallel
        # to _mode_term_cfgs["interval"], not the class-only subset.
        if "interval" in self._mode_term_cfgs:
            for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
                if not term_cfg.is_global_time:
                    assert term_cfg.interval_range_s is not None
                    lower, upper = term_cfg.interval_range_s
                    self._interval_term_time_left[index][ids] = self._sample_interval(
                        lower, upper, num_envs
                    )
        return {}

    def reset_mask(self, reset_mask: torch.Tensor) -> dict[str, float]:
        """Clear class-term state through the issue #2285 fixed-width mask."""
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                reset_mask_impl = getattr(term_cfg.func, "reset_mask", None)
                if not callable(reset_mask_impl):
                    raise NotImplementedError(
                        f"EventManager stateful term '{type(term_cfg.func).__name__}' "
                        "does not define the issue #2285 mask-preserving reset boundary"
                    )
                reset_mask_impl(reset_mask)
        return {}

    def apply(
        self,
        mode: EventMode,
        env_ids: torch.Tensor | slice | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
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
        if mode == "reset" and global_env_step_count is None:
            raise ValueError(
                f"Event mode '{mode}' requires the total number of environment steps to be provided."
            )
        if mode == "step" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")

        if mode == "reset" and self.uses_tensor_reset_rows:
            for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
                if term_cfg.min_step_count_between_reset:
                    assert global_env_step_count is not None
                    assert env_ids is not None
                    assert isinstance(env_ids, torch.Tensor)
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                    steps_since_triggered = global_env_step_count - last_triggered_step
                    valid_trigger = steps_since_triggered >= term_cfg.min_step_count_between_reset
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
                    valid_env_ids = env_ids[valid_trigger]
                    if valid_env_ids.numel() == 0:
                        continue
                    self._reset_term_last_triggered_once[index][valid_env_ids] = True
                    self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
                        global_env_step_count
                    )
                    term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
                    continue
                term_cfg.func(self._env, env_ids, **term_cfg.params)
            return

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
            elif mode == "reset":
                assert global_env_step_count is not None
                # Reset events require concrete indices: callers (e.g. ManagerBasedRlEnv)
                # resolve None to all environments upstream. Enforce that here so a future
                # caller passing None fails loudly instead of leaking a slice into event
                # functions, which only understand None or a tensor.
                if env_ids is None:
                    raise ValueError("Event mode 'reset' requires concrete environment indices.")
                min_step_count = term_cfg.min_step_count_between_reset
                if min_step_count == 0:
                    self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                    self._reset_term_last_triggered_once[index][env_ids] = True
                    term_cfg.func(self._env, env_ids, **term_cfg.params)
                else:
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                    steps_since_triggered = global_env_step_count - last_triggered_step
                    valid_trigger = steps_since_triggered >= min_step_count
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once
                    assert isinstance(env_ids, torch.Tensor)
                    valid_env_ids = env_ids[valid_trigger]
                    if valid_env_ids.numel() > 0:
                        self._reset_term_last_triggered_once[index][valid_env_ids] = True
                        self._reset_term_last_triggered_step_id[index][valid_env_ids] = (
                            global_env_step_count
                        )
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            else:
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    def apply_reset_mask(
        self,
        reset_mask: torch.Tensor,
        *,
        global_env_step_count: int,
    ) -> None:
        """Dispatch issue #2285's research-only fixed-shape reset events."""
        if reset_mask.ndim != 1 or reset_mask.shape[0] != self.num_envs:
            raise ValueError(f"EventManager research reset mask must have shape {(self.num_envs,)}")
        if reset_mask.dtype != torch.bool or reset_mask.device != self._device:
            raise ValueError(f"EventManager research reset mask must be bool on {self._device}")
        for index, term_cfg in enumerate(self._mode_term_cfgs["reset"]):
            term_func = getattr(term_cfg, "func", None)
            term_mask_impl = getattr(term_func, "reset_mask", None)
            if term_mask_impl is None:
                term_mask_impl = getattr(term_func, "__call_mask__", None)
            if not callable(term_mask_impl):
                raise NotImplementedError(
                    f"EventManager term '{self._mode_term_names['reset'][index]}' does not "
                    "define the issue #2285 mask-preserving reset boundary; func="
                    f"{getattr(term_func, '__name__', type(term_func).__name__)}"
                )
            min_step_count = term_cfg.min_step_count_between_reset
            term_mask = reset_mask
            if min_step_count:
                raise NotImplementedError(
                    f"EventManager reset term '{self._mode_term_names['reset'][index]}' "
                    "uses min_step_count_between_reset, which is not supported by the "
                    "issue #2285 mask transaction"
                )
            self._reset_term_last_triggered_step_id[index].masked_fill_(
                term_mask, global_env_step_count
            )
            self._reset_term_last_triggered_once[index].masked_fill_(term_mask, True)
            term_mask_impl(self._env, term_mask, **term_cfg.params)

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
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)
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
        return self._torch_rng.uniform(lower, upper, (count,), dtype=torch.float64).reshape(count)
