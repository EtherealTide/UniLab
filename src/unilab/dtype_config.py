"""Torch dtype owner for UniLab runtime carriers."""

from __future__ import annotations

import torch

# UniLab's Manager/tensor runtime owns float32 public carriers. Backend tensor
# specs negotiate any adapter-local storage; they do not mutate this owner.
GLOBAL_DTYPE = torch.float32


def get_global_dtype() -> torch.dtype:
    """Get the authoritative Torch dtype for environment runtime carriers."""
    return GLOBAL_DTYPE
