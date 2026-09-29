"""Device-resident rotation primitives shared by tensor task owners."""

from __future__ import annotations

import torch


def quat_mul(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    """Multiply broadcast-compatible wxyz quaternions without a host detour."""
    if left_wxyz.shape[-1] != 4 or right_wxyz.shape[-1] != 4:
        raise ValueError(
            f"quaternions must have final dimension 4; got {left_wxyz.shape} and {right_wxyz.shape}"
        )
    left_w, left_xyz = left_wxyz[..., 0:1], left_wxyz[..., 1:4]
    right_w, right_xyz = right_wxyz[..., 0:1], right_wxyz[..., 1:4]
    if left_xyz.ndim < right_xyz.ndim:
        left_xyz = left_xyz.expand_as(right_xyz)
    elif right_xyz.ndim < left_xyz.ndim:
        right_xyz = right_xyz.expand_as(left_xyz)
    cross = torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    return torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + cross,
        ),
        dim=-1,
    )


def quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Conjugate broadcast-compatible unit quaternions in wxyz order."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"quaternion must have final dimension 4; got {quat_wxyz.shape}")
    return torch.cat((quat_wxyz[..., 0:1], -quat_wxyz[..., 1:4]), dim=-1)


def quat_apply(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply broadcast-compatible wxyz quaternions to vectors."""
    if quat_wxyz.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError(
            f"quaternion/vector final dimensions must be (4, 3); got {quat_wxyz.shape} "
            f"and {vector.shape}"
        )
    quat_vector = quat_wxyz[..., 1:4]
    if vector.ndim < quat_vector.ndim:
        vector = vector.expand_as(quat_vector)
    cross = torch.linalg.cross(quat_vector, vector, dim=-1)
    second_cross = torch.linalg.cross(quat_vector, cross, dim=-1)
    return vector + 2.0 * quat_wxyz[..., 0:1] * cross + 2.0 * second_cross


def quat_apply_inverse(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse of broadcast-compatible unit quaternions."""
    return quat_apply(quat_conjugate(quat_wxyz), vector)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert broadcast-compatible XYZ Euler angles to wxyz quaternions."""
    if roll.shape != pitch.shape or roll.shape != yaw.shape:
        raise ValueError(
            f"Euler components must have matching shapes; got {roll.shape}, "
            f"{pitch.shape}, and {yaw.shape}"
        )
    cos_roll, sin_roll = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cos_pitch, sin_pitch = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cos_yaw, sin_yaw = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
        ),
        dim=-1,
    )


__all__ = [
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_from_euler_xyz",
    "quat_mul",
]
