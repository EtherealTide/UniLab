"""Dependency compatibility contracts owned by UniLab."""

from __future__ import annotations

from typing import Any

import unisim.support as support_module
from unisim.backend.base import TensorLifecycleCapabilities
from unisim.support import get_tensor_platform_profiles


def test_unisim_tensor_platform_boundaries_have_runtime_parity_guard() -> None:
    boundaries: dict[str, Any] = getattr(support_module, "_TENSOR_PLATFORM_BOUNDARIES", {})
    runtime_guard = getattr(support_module, "_assert_tensor_platform_capabilities", None)

    assert boundaries
    assert callable(runtime_guard)
    profiles = get_tensor_platform_profiles()

    for name, boundary in boundaries.items():
        profile = profiles[name]
        runtime = TensorLifecycleCapabilities(
            execution=boundary.execution,
            process_topology=boundary.process_topology,
            data_plane=boundary.data_plane,
            stream_event_ownership="reviewed dependency boundary",
            torch_devices=boundary.torch_devices,
        )

        assert callable(runtime_guard)
        runtime_guard(name, runtime)
        assert profile.execution is boundary.execution
        assert profile.process_topology is boundary.process_topology
        assert profile.data_plane is boundary.data_plane
        assert profile.torch_devices == boundary.torch_devices
