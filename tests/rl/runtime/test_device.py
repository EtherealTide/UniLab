"""Torch device alias resolution for the RL runtime (owner: rl.runtime.device).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15) ``tests/utils/test_device.py``.
Host hardware-info collection is owned by :mod:`unilab.utils.device` and
covered by ``tests/utils/test_device.py``; backend process-device routing is
owned by :mod:`unilab.base.process_device` and covered by
``tests/base/backend/test_process_device.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import unilab.rl.runtime.device as device_mod


def _patch_devices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    cuda_count: int = 0,
    xpu: bool = False,
    xpu_count: int = 0,
    mps: bool = False,
) -> None:
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(device_mod.torch.cuda, "device_count", lambda: cuda_count)
    monkeypatch.setattr(
        device_mod.torch,
        "xpu",
        SimpleNamespace(
            is_available=lambda: xpu,
            device_count=lambda: xpu_count,
        ),
        raising=False,
    )
    monkeypatch.setattr(device_mod.torch.backends.mps, "is_available", lambda: mps)


def test_resolve_torch_device_alias_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_devices(monkeypatch)

    assert device_mod.resolve_torch_device_alias(None) == "cpu"
    assert device_mod.resolve_torch_device_alias("cpu") == "cpu"


def test_resolve_torch_device_alias_gpu_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, cuda=True, cuda_count=2, mps=True)

    assert device_mod.resolve_torch_device_alias("gpu") == "cuda"
    assert device_mod.resolve_torch_device_alias("gpu:1") == "cuda:1"


def test_resolve_torch_device_alias_gpu_uses_xpu_before_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, xpu=True, xpu_count=2, mps=True)

    assert device_mod.resolve_torch_device_alias("gpu") == "xpu"
    assert device_mod.resolve_torch_device_alias("gpu:1") == "xpu:1"


@pytest.mark.parametrize("alias", ["gpu", "gpu:0"])
def test_resolve_torch_device_alias_macos_compat_maps_abstract_gpu_to_mps(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    _patch_devices(monkeypatch, mps=True)

    assert device_mod.resolve_torch_device_alias(alias) == "mps"


@pytest.mark.parametrize("alias", ["cuda", "cuda:0"])
def test_resolve_torch_device_alias_does_not_map_concrete_cuda_to_mps(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    _patch_devices(monkeypatch, mps=True)

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        device_mod.resolve_torch_device_alias(alias)


@pytest.mark.parametrize("alias", ["gpu:1", "mps:1"])
def test_resolve_torch_device_alias_mps_rejects_nonzero_index(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    _patch_devices(monkeypatch, mps=True)

    with pytest.raises(ValueError, match="MPS"):
        device_mod.resolve_torch_device_alias(alias)


def test_resolve_torch_device_alias_rejects_missing_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch, cuda=True, cuda_count=1)

    with pytest.raises(ValueError, match="only 1 cuda device"):
        device_mod.resolve_torch_device_alias("gpu:1")


def test_resolve_torch_device_alias_rejects_unavailable_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_devices(monkeypatch)

    with pytest.raises(ValueError, match="none is available"):
        device_mod.resolve_torch_device_alias("gpu")


def test_sync_device_uses_one_cuda_stream_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, torch.device]] = []

    class FakeStream:
        def synchronize(self) -> None:
            calls.append(("stream", torch.device("cuda:0")))

    monkeypatch.setattr(
        device_mod.torch.cuda,
        "current_stream",
        lambda device: FakeStream(),
    )
    device_mod.sync_device("cuda:0", stream=True)
    assert calls == [("stream", torch.device("cuda:0"))]


def test_device_runtime_capabilities_report_sync_and_timing_policy() -> None:
    assert device_mod.device_runtime_capabilities("cpu").manifest() == {
        "device_type": "cpu",
        "sync_api": "none",
        "stream_sync_api": None,
        "has_timing_events": False,
        "sync_policy": "capability",
        "timing_policy": "capability",
    }
    cuda = device_mod.device_runtime_capabilities("cuda")
    assert cuda.sync_api == "torch.cuda.synchronize"
    assert cuda.stream_sync_api == "torch.cuda.current_stream.synchronize"
    assert cuda.has_timing_events is True


def test_device_runtime_capabilities_fail_closed_for_unknown_devices() -> None:
    with pytest.raises((ValueError, RuntimeError), match="tpu"):
        device_mod.device_runtime_capabilities("tpu")


def test_synchronized_device_timer_uses_wall_clock_boundary() -> None:
    timer = device_mod.SynchronizedDeviceTimer("cpu")
    timer.start()
    timer.stop()
    assert timer.elapsed_s() >= 0.0
