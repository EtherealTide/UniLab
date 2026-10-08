"""Render-only visual model override for the MuJoCo scene context.

These cover the cold-path wiring that lets offline playback render a visual twin
of a scene (e.g. a per-env replicable wall) without touching the trained physics
model. See ``SceneCfg.visual_model_file`` and ``_build_mujoco_scene_context``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjbatch")
pytest.importorskip(
    "unisim.backend.mujoco.backend",
    reason="unisim-core MuJoCo adapter (mjbatch build) not available",
)

from unisim.backend.mujoco.backend import _build_mujoco_scene_context

from unilab.base.scene import SceneCfg


def test_visual_model_file_defaults_to_model_file() -> None:
    ctx = _build_mujoco_scene_context(SceneCfg(model_file="/tmp/phys.xml"))

    assert ctx.model_file == "/tmp/phys.xml"
    assert ctx.model_source == "/tmp/phys.xml"
    # Unset override => renderer uses the physics model (behaviour unchanged for
    # every task that does not opt in).
    assert ctx.visual_model_file == "/tmp/phys.xml"


def test_visual_model_file_override_is_render_only() -> None:
    ctx = _build_mujoco_scene_context(
        SceneCfg(model_file="/tmp/phys.xml", visual_model_file="/tmp/visual.xml")
    )

    # Physics keeps using model_file; only the render model is swapped.
    assert ctx.model_file == "/tmp/phys.xml"
    assert ctx.model_source == "/tmp/phys.xml"
    assert ctx.visual_model_file == "/tmp/visual.xml"
