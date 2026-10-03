"""Compiled tensor root-reset kernel contracts."""

from __future__ import annotations

import torch

from unilab.envs.mdp.events import _tensor_root_reset_kernel


def test_root_reset_kernel_applies_pose_velocity_and_origins() -> None:
    default = torch.zeros((2, 13), dtype=torch.float32)
    default[:, 3] = 1.0
    origins = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).repeat(2, 1)
    pose = torch.zeros((2, 6), dtype=torch.float32)
    pose[:, 0] = 0.5
    pose[:, 5] = torch.pi / 2
    velocity = torch.zeros((2, 6), dtype=torch.float32)
    velocity[:, 0] = 1.5
    output = torch.empty_like(default)

    _tensor_root_reset_kernel(default, origins, pose, velocity, output)

    torch.testing.assert_close(
        default,
        torch.zeros_like(default)
        + torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    )
    torch.testing.assert_close(output[:, 0], torch.full((2,), 1.5))
    torch.testing.assert_close(output[:, 1], torch.full((2,), 2.0))
    torch.testing.assert_close(output[:, 7], torch.full((2,), 1.5))
    torch.testing.assert_close(
        output[:, 3], torch.full((2,), torch.tensor(0.70710678118)), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        output[:, 6], torch.full((2,), torch.tensor(0.70710678118)), rtol=0, atol=1e-6
    )


def test_root_reset_kernel_accepts_disabled_origins() -> None:
    default = torch.zeros((3, 13), dtype=torch.float32)
    default[:, 3] = 1.0
    pose = torch.zeros((3, 6), dtype=torch.float32)
    pose[:, 1] = 0.25
    velocity = torch.zeros((3, 6), dtype=torch.float32)
    output = torch.empty_like(default)

    _tensor_root_reset_kernel(default, None, pose, velocity, output)

    torch.testing.assert_close(output[:, 1], torch.full((3,), 0.25))
    torch.testing.assert_close(output[:, 3], torch.ones(3))
