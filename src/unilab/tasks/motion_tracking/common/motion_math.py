"""Tensor-native motion math shared by Manager-owned terms and tests."""

from __future__ import annotations

import torch


def _gravity_z_in_body(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2) - 1.0


def _adaptive_failure_counts(
    bin_indices: torch.Tensor, terminated: torch.Tensor, n_bins: int
) -> torch.Tensor:
    discard_bin = n_bins
    failed_bins = torch.where(
        terminated,
        bin_indices,
        torch.full_like(bin_indices, discard_bin),
    )
    return torch.bincount(failed_bins, minlength=discard_bin + 1)[:n_bins].to(torch.float32)


def _adaptive_failure_alpha(terminated: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.any(terminated).to(dtype=torch.float32) * alpha
