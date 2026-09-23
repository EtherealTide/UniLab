"""Tests for the contract-driven physics-state applier and startup precheck.

Covers unilabsim/unisim#291 downstream consumption: capability precheck at
entrypoint startup, layout-contract snapshot splitting, and contract mocap
replay.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from unisim.backend.base import PhysicsStateLayout

from unilab.base.base import EnvPlayCapabilities
from unilab.visualization.playback_state import (
    PhysicsStateApplier,
    assert_physics_state_playback_supported,
)

_FREE_BODY_XML = """
<mujoco>
  <worldbody>
    <body><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""

_MOCAP_XML = """
<mujoco>
  <worldbody>
    <body mocap="true"><geom type="sphere" size="0.05"/></body>
    <body><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


class _StubBackend:
    pass


class _StubEnv:
    """Minimal env surface consumed by the playback-state helpers."""

    def __init__(
        self,
        capabilities: EnvPlayCapabilities,
        layout: PhysicsStateLayout,
    ) -> None:
        self.play_capabilities = capabilities
        self._backend = _StubBackend()
        self._layout = layout
        self.mocap_calls: list[int] = []

    def get_physics_state_layout(self) -> Any:
        return self._layout

    def get_playback_mocap_state(self, env_index: int = 0):
        self.mocap_calls.append(env_index)
        return np.full((1, 3), 0.5), np.array([[1.0, 0.0, 0.0, 0.0]])


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def test_precheck_passes_with_physics_state_playback() -> None:
    env = _StubEnv(
        EnvPlayCapabilities(supports_physics_state_playback=True),
        layout=PhysicsStateLayout(nq=0, nv=0),
    )
    assert_physics_state_playback_supported(env, entrypoint="play_viser")


def test_precheck_names_entrypoint_backend_and_native_hint() -> None:
    env = _StubEnv(
        EnvPlayCapabilities(supports_native_interactive_renderer=True),
        layout=PhysicsStateLayout(nq=0, nv=0),
    )
    with pytest.raises(
        NotImplementedError, match=r"play_interactive.*_StubBackend.*native interactive"
    ):
        assert_physics_state_playback_supported(env, entrypoint="play_interactive")


def test_precheck_points_to_issue_when_no_rendering_path() -> None:
    env = _StubEnv(EnvPlayCapabilities(), layout=PhysicsStateLayout(nq=0, nv=0))
    with pytest.raises(NotImplementedError, match=r"play_viser.*unisim#291"):
        assert_physics_state_playback_supported(env, entrypoint="play_viser")


def test_applier_splits_time_qpos_qvel() -> None:
    model = _model(_FREE_BODY_XML)
    env = _StubEnv(
        EnvPlayCapabilities(supports_physics_state_playback=True),
        layout=PhysicsStateLayout(nq=7, nv=6),
    )
    applier = PhysicsStateApplier(env, model, env_index=0)

    data = mujoco.MjData(model)
    row = np.arange(14, dtype=np.float64)
    applier.apply(row, data)

    assert data.time == pytest.approx(0.0)
    np.testing.assert_allclose(data.qpos, row[1:8])
    np.testing.assert_allclose(data.qvel, row[8:14])


def test_applier_replays_mocap_through_contract_entrypoint() -> None:
    model = _model(_MOCAP_XML)
    assert model.nmocap == 1
    env = _StubEnv(
        EnvPlayCapabilities(supports_physics_state_playback=True, supports_mocap_playback=True),
        layout=PhysicsStateLayout(nq=7, nv=6, nmocap=1),
    )
    applier = PhysicsStateApplier(env, model, env_index=3)

    data = mujoco.MjData(model)
    applier.apply(np.zeros(1 + 7 + 6 + 7, dtype=np.float64), data)

    assert env.mocap_calls == [3]
    np.testing.assert_allclose(data.mocap_pos, 0.5)
    np.testing.assert_allclose(data.mocap_quat, [[1.0, 0.0, 0.0, 0.0]])


def test_applier_falls_back_to_snapshot_tail_without_mocap_capability() -> None:
    model = _model(_MOCAP_XML)
    env = _StubEnv(
        EnvPlayCapabilities(supports_physics_state_playback=True),
        layout=PhysicsStateLayout(nq=7, nv=6, nmocap=1),
    )
    applier = PhysicsStateApplier(env, model)

    data = mujoco.MjData(model)
    row = np.arange(21, dtype=np.float64)
    applier.apply(row, data)

    assert env.mocap_calls == []
    np.testing.assert_allclose(data.mocap_pos[0], row[14:17])
    np.testing.assert_allclose(data.mocap_quat[0], row[17:21])


def test_applier_rejects_model_layout_mismatch() -> None:
    model = _model(_FREE_BODY_XML)
    env = _StubEnv(
        EnvPlayCapabilities(supports_physics_state_playback=True),
        layout=PhysicsStateLayout(nq=8, nv=6),
    )
    with pytest.raises(ValueError, match="nq=8"):
        PhysicsStateApplier(env, model)
