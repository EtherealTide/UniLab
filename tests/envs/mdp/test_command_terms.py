"""Contract tests for generic pose command terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.envs.mdp import UniformPoseCommand, UniformPoseCommandCfg
from unilab.managers.rng import ManagerRng


def _env(num_envs: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        rng=ManagerRng.seeded(11),
    )


def test_uniform_pose_command_samples_configured_width_and_zero_bucket() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=((-1.0, 1.0), (2.0, 2.0)),
        zero_command_prob=1.0,
    )
    term = cfg.build(env)
    assert isinstance(term, UniformPoseCommand)

    term.reset(torch.ones(2, dtype=torch.bool))
    assert term.command.shape == (2, 2)
    np.testing.assert_array_equal(term.command, 0.0)


def test_uniform_pose_command_reads_curriculum_updates_without_changing_width() -> None:
    env = _env()
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(1.0, 1.0),
        ranges=[[0.0, 0.0], [1.0, 1.0]],
    )
    term = cfg.build(env)
    reset_mask = torch.ones(2, dtype=torch.bool)

    cfg.ranges[0] = [2.0, 2.0]
    term.reset(reset_mask)
    np.testing.assert_array_equal(term.command, [[2.0, 1.0], [2.0, 1.0]])

    cfg.ranges.append([3.0, 3.0])
    with pytest.raises(ValueError, match="ranges width changed"):
        term.reset(reset_mask)


def test_uniform_pose_command_fails_closed_without_rng() -> None:
    env = _env()
    env.rng = None

    with pytest.raises(RuntimeError, match="CommandTerm scheduling"):
        UniformPoseCommandCfg(
            resampling_time_range=(1.0, 1.0),
            ranges=((0.0, 1.0),),
        ).build(env)
