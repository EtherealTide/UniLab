"""Data-parallel rank topology context tests (owner: rl.runtime.distributed.context).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/ipc/test_dp_launcher.py`` and UniLab's former
``tests/ipc/test_dp_launcher.py``.
"""

from __future__ import annotations

import pytest
import torch

import unilab.rl.runtime.distributed.context as context
from unilab.rl.runtime.distributed.context import (
    UNILAB_DP_RANK,
    UNILAB_DP_WORLD_SIZE,
    current_dp_rank,
    current_dp_world_size,
    current_torch_distributed_local_rank,
    current_torch_distributed_rank,
    current_torch_distributed_world_size,
    rank_local_cuda_device,
    selected_visible_entries,
    visible_cuda_entries,
)


def test_current_dp_rank_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.delenv(UNILAB_DP_WORLD_SIZE, raising=False)
    assert current_dp_rank() == 0
    assert current_dp_world_size() == 1


def test_current_dp_rank_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(UNILAB_DP_RANK, "2")
    monkeypatch.setenv(UNILAB_DP_WORLD_SIZE, "4")
    assert current_dp_rank() == 2
    assert current_dp_world_size() == 4


def test_current_torch_distributed_rank_defaults(monkeypatch: pytest.MonkeyPatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    assert current_torch_distributed_rank() == 0
    assert current_torch_distributed_local_rank() == 0
    assert current_torch_distributed_world_size() == 1


def test_current_torch_distributed_rank_reads_torchrun_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert current_torch_distributed_rank() == 3
    assert current_torch_distributed_local_rank() == 1
    assert current_torch_distributed_world_size() == 4


def test_visible_cuda_entries_accepts_opaque_rank_local_tokens() -> None:
    assert visible_cuda_entries(None) == ()
    assert visible_cuda_entries("") == ()
    assert visible_cuda_entries("-1") == ()
    assert visible_cuda_entries("0") == ("0",)
    assert visible_cuda_entries("GPU-AAAA, MIG-BBBB") == ("GPU-AAAA", "MIG-BBBB")


def test_single_visible_gpu_is_rank_local_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-7")

    assert rank_local_cuda_device() == "cuda:0"


def test_rank_device_uses_single_visibility_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv(context.UNILAB_DP_WORLD_SIZE, raising=False)

    assert rank_local_cuda_device() == "cuda:0"


def test_selected_visibility_preserves_parent_entries(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    assert selected_visible_entries(world_size=3) == ("GPU-a", "GPU-b", "GPU-c")


def test_selected_visibility_requires_exact_rank_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(ValueError, match="one entry per data-parallel rank"):
        selected_visible_entries(world_size=3)


def test_rank_local_cuda_device_rejects_multiple_visible_entries(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    with pytest.raises(ValueError, match="exactly one CUDA_VISIBLE_DEVICES"):
        rank_local_cuda_device(rank=1)


def test_unset_mask_with_one_visible_cuda_device_is_rank_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert rank_local_cuda_device() == "cuda:0"


def test_unset_mask_with_multiple_visible_cuda_devices_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert rank_local_cuda_device() is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices")
@pytest.mark.slow
def test_rank_local_visibility_validates_on_two_gpu_host(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    assert selected_visible_entries(world_size=2) == ("GPU-a", "GPU-b")
