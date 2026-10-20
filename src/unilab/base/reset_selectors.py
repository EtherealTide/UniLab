"""Reset row-selector conversion between the public ``env_ids`` and mask forms.

The task/manager reset contract speaks IsaacLab/mjlab-style ``env_ids`` row
indices. Backend transactions keep consuming fixed-width bool masks
(``SimBackend.apply_reset``/packed reset). These helpers are the single,
deliberately simple conversion point between the two forms; the optimized
adaptation layer is tracked separately (roadmap #2461).
"""

from __future__ import annotations

import warnings

import torch

_MASK_DEPRECATION = (
    "{owner} received a fixed-width bool reset mask; the public reset contract "
    "takes env_ids (row indices) again. Mask selectors are deprecated since "
    "v1.5 and will be removed in v1.6."
)


def resolve_reset_env_ids(
    selector: torch.Tensor | slice | None,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> torch.Tensor:
    """Normalize a public reset selector to an int64 row-index tensor.

    ``None`` selects every row; a slice selects ``arange(num_envs)[slice]``; an
    integer tensor is validated and moved onto ``device``. A fixed-width bool
    mask is the deprecated form: it is accepted for one release with a
    ``DeprecationWarning`` and converted to row indices.
    """
    if selector is None:
        return torch.arange(num_envs, dtype=torch.int64, device=device)
    if isinstance(selector, slice):
        return torch.arange(num_envs, dtype=torch.int64, device=device)[selector]
    if not isinstance(selector, torch.Tensor):
        raise TypeError(
            f"{owner} env_ids must be a torch.Tensor, slice, or None; got {type(selector).__name__}"
        )
    if selector.dtype == torch.bool:
        if selector.shape != (num_envs,):
            raise ValueError(
                f"{owner} reset mask must have shape {(num_envs,)}; got {tuple(selector.shape)}"
            )
        warnings.warn(_MASK_DEPRECATION.format(owner=owner), DeprecationWarning, stacklevel=2)
        return torch.nonzero(selector, as_tuple=False).flatten().to(device=device)
    if selector.ndim != 1:
        raise ValueError(
            f"{owner} env_ids must be a 1-D row-index tensor; got shape {tuple(selector.shape)}"
        )
    ids = selector.to(device=device, dtype=torch.int64)
    if ids.numel():
        if bool((ids < 0).any()) or bool((ids >= num_envs).any()):
            raise ValueError(f"{owner} env_ids contain rows outside [0, {num_envs})")
    return ids


def reset_env_ids_mask(
    env_ids: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter row indices into the fixed-width bool backend transaction mask."""
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if env_ids.numel():
        mask[env_ids.to(device=device, dtype=torch.int64)] = True
    return mask


def resolve_reset_mask(
    selector: torch.Tensor | slice | None,
    *,
    num_envs: int,
    device: torch.device,
    owner: str,
) -> torch.Tensor:
    """Resolve a public reset selector to the internal fixed-width bool mask.

    A bool mask is the deprecated form: it passes through for one release with
    a ``DeprecationWarning``. Everything else is normalized to ``env_ids`` and
    scattered into the mask.
    """
    if (
        isinstance(selector, torch.Tensor)
        and selector.dtype == torch.bool
        and selector.shape == (num_envs,)
    ):
        warnings.warn(_MASK_DEPRECATION.format(owner=owner), DeprecationWarning, stacklevel=2)
        return selector.to(device=device)
    env_ids = resolve_reset_env_ids(selector, num_envs=num_envs, device=device, owner=owner)
    return reset_env_ids_mask(env_ids, num_envs=num_envs, device=device)


__all__ = [
    "reset_env_ids_mask",
    "resolve_reset_env_ids",
    "resolve_reset_mask",
]
