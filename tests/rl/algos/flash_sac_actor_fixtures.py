"""Importable custom FlashSAC actors shared by the custom-actor tests.

The builders are resolved by dotted path (``tests.rl.algos...:build_*``) the
same way downstream packages expose theirs, and this module is importable in
spawn subprocesses, so it doubles as the spawn-safety fixture.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_tanh_log_det(raw: torch.Tensor, scale: float) -> torch.Tensor:
    """Per-dim log|det J| of ``scale * tanh(raw)`` (numerically stable)."""
    return math.log(scale) + 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))


class ScaledTanhToyActor(nn.Module):
    """Toy custom actor with a scaled-tanh action transform and one aux loss."""

    action_transform_id = "scaled_tanh_test_v1"

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
        hidden_dim: int = 16,
        action_scale: float = 2.0,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.aux_weight = float(aux_weight)
        self.emit_aux = True
        self.trunk = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.Tanh())
        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        self.aux_head = nn.Linear(self.obs_dim, self.action_dim)
        self.to(device)

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del training
        hidden = self.trunk(observations)
        mean = self.mean_head(hidden)
        std = F.softplus(self.log_std_head(hidden)) + 1e-4
        return mean, std

    def log_prob_from_raw(
        self, mean: torch.Tensor, std: torch.Tensor, raw: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw) - _scaled_tanh_log_det(raw, self.action_scale)
        return log_prob.sum(dim=-1)

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mean, std = self.get_mean_and_std(observations, training)
        raw = torch.distributions.Normal(mean, std).rsample()
        actions = self.action_scale * torch.tanh(raw)
        info: dict[str, Any] = {
            "log_prob": self.log_prob_from_raw(mean, std, raw),
            "mean": mean,
            "std": std,
        }
        if training and self.emit_aux:
            reconstruction = self.aux_head(observations)
            info["aux_losses"] = {
                "token_recon": self.aux_weight * ((reconstruction - actions.detach()) ** 2).mean()
            }
        return actions, info

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del dones
        mean, std = self.get_mean_and_std(obs, training=False)
        if deterministic:
            return self.action_scale * torch.tanh(mean)
        return self.action_scale * torch.tanh(mean + std * torch.randn_like(mean))

    def normalize_parameters(self) -> None:
        return None

    def as_export_module(self) -> nn.Module:
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                mean, _std = self.base.get_mean_and_std(obs, training=False)
                return actor.action_scale * torch.tanh(mean)

        return _Wrapper()

    def inference_startup_memory_categories(self, batch_size: int) -> dict[str, int]:
        return {"persistent_exploration_scratch": 0}


class BrokenToyActor(ScaledTanhToyActor):
    """Missing the exploration hook; contract validation must reject it."""

    explore = None  # type: ignore[assignment]


class BadAuxToyActor(ScaledTanhToyActor):
    """Emits a non-scalar auxiliary loss; contract validation must reject it."""

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions, info = super().forward(observations, training)
        if training:
            info["aux_losses"] = {"bad": torch.zeros(self.action_dim)}
        return actions, info


def build_scaled_tanh_actor(
    *,
    obs_dim: int,
    action_dim: int,
    device: str | torch.device,
    **kwargs: Any,
) -> ScaledTanhToyActor:
    return ScaledTanhToyActor(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)


def build_broken_actor(
    *,
    obs_dim: int,
    action_dim: int,
    device: str | torch.device,
    **kwargs: Any,
) -> BrokenToyActor:
    return BrokenToyActor(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)


def build_bad_aux_actor(
    *,
    obs_dim: int,
    action_dim: int,
    device: str | torch.device,
    **kwargs: Any,
) -> BadAuxToyActor:
    return BadAuxToyActor(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)


def build_non_module_actor(**kwargs: Any) -> object:
    del kwargs
    return object()


def spawn_build_child(
    builder_path: str,
    kwargs_json: str,
    obs_dim: int,
    action_dim: int,
    result_queue: Any,
) -> None:
    """Spawn-subprocess entry point: rebuild the actor from its spec."""
    from unilab.rl.algos.flash_sac.actor_spec import FlashSACActorSpec, build_custom_actor

    spec = FlashSACActorSpec(builder=builder_path, kwargs=json.loads(kwargs_json))
    actor = build_custom_actor(spec, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    obs = torch.zeros(3, obs_dim)
    deterministic = actor.explore(obs, deterministic=True)
    exported = actor.as_export_module()(obs)
    result_queue.put(
        {
            "max_abs": float(deterministic.abs().max()),
            "export_matches": bool(torch.allclose(exported, deterministic)),
            "transform_id": actor.action_transform_id,
        }
    )
