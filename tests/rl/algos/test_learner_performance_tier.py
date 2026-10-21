"""Learner performance-tier policy contract."""

from __future__ import annotations

import pytest
import torch

from unilab.rl.algos.common.performance import resolve_learner_performance_tier


def test_nvidia_cuda_is_the_validated_accelerated_tier() -> None:
    tier = resolve_learner_performance_tier("cuda:0")

    assert tier.name == "nvidia_cuda"
    assert tier.vendor_runtime == "nvidia"
    assert tier.supports_whole_cycle_graph is True
    assert tier.requires_inductor is True
    assert tier.supports_capturable_optimizer is True
    assert tier.evidence == "policy"


def test_rocm_cuda_remains_compatibility_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)

    tier = resolve_learner_performance_tier("cuda:0")

    assert tier.name == "rocm_cuda_compatibility"
    assert tier.vendor_runtime == "rocm"
    assert not tier.supports_whole_cycle_graph
    assert not tier.requires_inductor
    assert not tier.supports_capturable_optimizer


def test_mps_cpu_and_xpu_use_compatibility_tiers() -> None:
    for device_type in ("mps", "cpu", "xpu"):
        tier = resolve_learner_performance_tier(device_type)
        assert tier.name == f"{device_type}_compatibility"
        assert not tier.supports_whole_cycle_graph
        assert not tier.requires_inductor
        assert not tier.supports_capturable_optimizer


def test_unknown_learner_device_fails_closed() -> None:
    with pytest.raises((ValueError, RuntimeError), match="CUDA, MPS, XPU, and CPU"):
        resolve_learner_performance_tier("tpu")


def test_tier_manifest_separates_policy_from_capability_language() -> None:
    manifest = resolve_learner_performance_tier("cuda:0").manifest()
    assert manifest["evidence"] == "policy"
    assert manifest == {
        "name": "nvidia_cuda",
        "device_type": "cuda",
        "vendor_runtime": "nvidia",
        "supports_whole_cycle_graph": True,
        "requires_inductor": True,
        "supports_capturable_optimizer": True,
        "evidence": "policy",
    }
