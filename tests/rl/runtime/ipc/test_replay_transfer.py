"""Replay transfer capability policy and factory contract."""

from __future__ import annotations

import pytest
import torch

import unilab.rl.runtime.ipc.replay_pipelines.native_h2d as native_h2d
import unilab.rl.runtime.ipc.replay_pipelines.transfer.cuda_like as cuda_like_module
from unilab.rl.runtime.ipc.replay_pipelines.transfer import build_replay_transfer_backend
from unilab.rl.runtime.ipc.replay_pipelines.transfer.torch_copy import (
    TorchCopyReplayTransferBackend,
)


def _patch_cudart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: type(
            "FakeCudart",
            (),
            {
                "cudaError": type("Errors", (), {"success": 0}),
            },
        )(),
    )


def test_replay_transfer_factory_fails_closed_for_unsupported_devices() -> None:
    with pytest.raises(ValueError, match="CUDA or MPS"):
        build_replay_transfer_backend(device=torch.device("cpu"), ring_depth=2)


def test_torch_copy_backend_declares_portable_capabilities() -> None:
    backend = TorchCopyReplayTransferBackend(device=torch.device("mps"), ring_depth=2)

    assert backend.device_family == "mps"
    assert backend.h2d_submitter == "torch_copy"
    assert backend.supports_pinned is False
    assert backend.supports_timing_events is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend required")
def test_cuda_like_backend_selects_native_h2d_by_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cudart(monkeypatch)
    monkeypatch.setattr(native_h2d, "is_available", lambda: True)

    class FakeVersion:
        hip = "6.3"
        cuda = "12.8"

    monkeypatch.setattr(torch, "version", FakeVersion(), raising=False)
    monkeypatch.setattr(cuda_like_module, "is_available", lambda: True)
    backend = cuda_like_module.CudaLikeReplayTransferBackend(
        device=torch.device("cuda"), ring_depth=2
    )

    assert backend.device_family == "rocm"
    assert backend.h2d_submitter == "pybind11"
    assert backend.native_h2d_available is True
