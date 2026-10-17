"""Isaac external-worker owner-option routing."""

from __future__ import annotations

from unilab.base.backend_factory import env_backend_kwargs
from unilab.base.base import EnvCfg


def test_env_backend_kwargs_maps_isaac_owner_options() -> None:
    cfg = EnvCfg(
        isaacgym_device_id=1,
        isaacgym_worker_timeout_s=120.0,
        isaacsim_device_id=2,
        isaacsim_worker_timeout_s=300.0,
        isaacsim_render_mode="none",
        isaacsim_render_width=640,
        isaacsim_render_height=480,
        isaacsim_tensor_cuda_ipc=True,
        isaacsim_share_friction_materials=True,
    )
    kwargs = env_backend_kwargs(cfg)

    assert kwargs["isaacgym_device_id"] == 1
    assert kwargs["isaacgym_worker_timeout_s"] == 120.0
    assert kwargs["isaacsim_device_id"] == 2
    assert kwargs["isaacsim_worker_timeout_s"] == 300.0
    assert kwargs["isaacsim_render_mode"] == "none"
    assert kwargs["isaacsim_render_width"] == 640
    assert kwargs["isaacsim_render_height"] == 480
    assert kwargs["isaacsim_tensor_cuda_ipc"] is True
    assert kwargs["share_friction_materials"] is True


def test_isaac_opt_in_keys_are_not_sent_when_disabled() -> None:
    kwargs = env_backend_kwargs(EnvCfg())

    assert "isaacsim_tensor_cuda_ipc" not in kwargs
    assert "share_friction_materials" not in kwargs
