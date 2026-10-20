"""Playback-mode camera degradation tests for the ABEnv playback seam."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest
from unisim.backend.base import CameraCfg, PlayRenderPlan

from unilab.base.base import ABEnv, EnvCfg


class _StubPlaybackEnv(ABEnv):
    """Minimal ABEnv driving run_playback_mode with a scripted run_playback."""

    def __init__(self, plan: PlayRenderPlan):
        self._plan = plan
        self.calls: list[dict[str, Any]] = []

    def resolve_play_render_plan(self, **kwargs: Any) -> PlayRenderPlan:
        return self._plan

    def run_playback(self, **kwargs: Any) -> str | None:
        self.calls.append(kwargs)
        camera_kwargs = kwargs.get("camera_kwargs")
        if camera_kwargs is not None:
            raise NotImplementedError(
                "Isaac native interactive viewers do not apply camera_kwargs; "
                "configure the view through the native viewer."
            )
        return None

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def cfg(self) -> EnvCfg:
        return EnvCfg()

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict()

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict()

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {}

    @property
    def state(self) -> Any:
        return None

    def init_state(self) -> Any:
        return None

    def step(self, actions: Any) -> Any:
        return None

    def close(self) -> None:
        pass


def _interactive_plan() -> PlayRenderPlan:
    return PlayRenderPlan(
        mode="interactive",
        num_steps=None,
        output_video=None,
        headless=False,
        record_video=False,
    )


def test_interactive_playback_drops_rejected_camera_kwargs() -> None:
    env = _StubPlaybackEnv(_interactive_plan())
    camera = CameraCfg.from_kwargs({"cam_distance": 6.0})

    with pytest.warns(UserWarning, match="camera_kwargs"):
        result = env.run_playback_mode(
            play_render_mode="interactive",
            play_steps=None,
            output_video=None,
            initialize=lambda: "obs_0",
            step=lambda obs: obs,
            camera_kwargs=camera,
        )

    assert result is None
    assert len(env.calls) == 2
    assert env.calls[0]["camera_kwargs"] == camera
    assert env.calls[1]["camera_kwargs"] is None


def test_interactive_playback_without_camera_kwargs_runs_directly() -> None:
    env = _StubPlaybackEnv(_interactive_plan())

    result = env.run_playback_mode(
        play_render_mode="interactive",
        play_steps=None,
        output_video=None,
        initialize=lambda: "obs_0",
        step=lambda obs: obs,
    )

    assert result is None
    assert len(env.calls) == 1
    assert env.calls[0]["camera_kwargs"] is None


def test_record_playback_camera_rejection_still_fails_closed() -> None:
    plan = PlayRenderPlan(
        mode="record",
        num_steps=4,
        output_video="play_video.mp4",
        headless=True,
        record_video=True,
    )
    env = _StubPlaybackEnv(plan)

    with pytest.raises(NotImplementedError, match="camera_kwargs"):
        env.run_playback_mode(
            play_render_mode="record",
            play_steps=4,
            output_video="play_video.mp4",
            initialize=lambda: "obs_0",
            step=lambda obs: obs,
            camera_kwargs=CameraCfg.from_kwargs({"cam_distance": 6.0}),
        )

    assert len(env.calls) == 1
