"""Fail-closed public capability matrix for the ``mjwarp`` host profile."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from unisim.backend.mjwarp.dependencies import load_mjwarp_dependencies
from unisim.dr.types import IntervalRandomizationPlan

from unilab.base.backend_factory import create_backend
from unilab.base.scene import SceneCfg

pytestmark = pytest.mark.slow


def _require_cuda_mjwarp() -> None:
    if sys.platform == "darwin":
        pytest.skip("mjwarp is a CUDA-only backend; macOS has no supported CUDA runtime")
    dependencies = load_mjwarp_dependencies()
    if not bool(dependencies.warp.get_device().is_cuda):
        pytest.fail("mjwarp capability tests require an active CUDA Warp device")


def _backend() -> Any:
    _require_cuda_mjwarp()

    from unilab.assets import ASSETS_ROOT_PATH

    scene = SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))
    return create_backend("mjwarp", scene, 1, 0.02 / 3.0, base_name="pelvis")


def test_unsupported_matrix_fails_before_step() -> None:
    """Every currently unadvertised public path errors before a physics step."""
    backend = _backend()
    with pytest.raises(NotImplementedError, match="body positions"):
        backend.get_body_pos_w(np.asarray([1], dtype=np.int32))
    with pytest.raises(NotImplementedError, match="height-field scanners"):
        backend.create_hfield_scanner(
            hfield_geom_id=0,
            offsets=np.zeros((1, 2), dtype=np.float32),
            frame_body_id=1,
        )
    assert backend.get_play_capabilities().supports_physics_state_playback
    assert not backend.get_play_capabilities().supports_native_interactive_renderer
    assert not backend.get_play_capabilities().supports_native_video_capture
    assert Path(backend.get_playback_model()).is_file()
    # Current unisim-core exposes per-world gravity reset randomization.
    capabilities = backend.get_dr_capabilities()
    assert capabilities.supports_reset_term("gravity")


def test_interval_push_and_velocity_require_named_bodies() -> None:
    """Without base/push body names the interval capabilities stay fail-closed."""
    _require_cuda_mjwarp()

    from unilab.assets import ASSETS_ROOT_PATH

    scene = SceneCfg(model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"))
    backend = create_backend("mjwarp", scene, 1, 0.02 / 3.0)
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supports_interval_push
    assert not capabilities.supports_interval_body_velocity_delta
    with pytest.raises(NotImplementedError, match="push target body"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(push_perturbation_limit=np.ones((3,), dtype=np.float32))
        )
    with pytest.raises(NotImplementedError, match="exactly one free joint"):
        backend.apply_interval_randomization(
            IntervalRandomizationPlan(
                body_ids=np.asarray([1], dtype=np.int32),
                body_linear_velocity_delta=np.zeros((1, 1, 3), dtype=np.float32),
            )
        )
    with pytest.raises(ValueError, match="Push body 'missing' not found"):
        create_backend("mjwarp", scene, 1, 0.02 / 3.0, push_body_name="missing")
