"""Pickleable runtime hooks injected across the ``uni_rl`` spawn boundary.

``uni_rl`` owns collector orchestration but must not import UniLab.  The
off-policy runner passes these module-level constructors to the collector so
environment-owned diagnostics stay in their owner package.
"""

from __future__ import annotations

from typing import Any


def tensor_nan_guard_factory(
    cfg: Any,
    num_envs: int,
    supports_state_playback: bool,
) -> Any:
    """Construct UniLab's device-resident tensor finite-state guard."""
    from unilab.training.tensor_diagnostics import TensorNanGuard

    return TensorNanGuard(
        cfg,
        num_envs=num_envs,
        supports_state_playback=supports_state_playback,
    )


def build_tensor_nan_guard_factory() -> Any:
    """Return the pickleable constructor used by spawn collectors."""
    return tensor_nan_guard_factory
