"""Unit tests for the PPO/APPO TF32 matmul owner module (#2475)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from unilab.rl.runtime.tf32 import (
    DEFAULT_TF32_MATMUL,
    TF32_MATMUL_VALUES,
    Tf32Policy,
    apply_tf32_matmul,
    apply_tf32_matmul_from_cfg,
    format_tf32_matmul,
    normalize_tf32_matmul,
    resolve_tf32_matmul,
    resolve_tf32_matmul_from_cfg,
)

_CONF_DIR = Path(__file__).resolve().parents[3] / "src" / "unilab" / "conf"


class _FakeCuda:
    def __init__(self, *, available: bool = True, capability: tuple[int, int] = (8, 9)):
        self.available = available
        self.capability = capability

    def is_available(self) -> bool:
        return self.available

    def set_device(self, device: Any = None) -> None:
        del device

    def get_device_capability(self, device: Any = None) -> tuple[int, int]:
        del device
        return self.capability


def _fake_torch(
    *,
    hip: str | None = None,
    cuda: _FakeCuda | None = None,
    precision: str = "none",
):
    class _Matmul:
        def __init__(self) -> None:
            self.fp32_precision = precision
            self.allow_tf32 = precision == "tf32"

    class _Cudnn:
        def __init__(self) -> None:
            self.fp32_precision = precision
            self.allow_tf32 = precision == "tf32"

    class _Backends:
        def __init__(self) -> None:
            self.cuda = type("_CudaBackends", (), {"matmul": _Matmul()})()
            self.cudnn = _Cudnn()

    torch_like = type("_Torch", (), {})()
    torch_like.version = type("_V", (), {"hip": hip})()
    if cuda is not None:
        torch_like.cuda = cuda
    torch_like.backends = _Backends()
    torch_like.device = lambda value=None: type(
        "_Device", (), {"type": str(value).split(":", 1)[0] if value else "cpu"}
    )()
    return torch_like


def test_only_two_values_are_accepted():
    assert TF32_MATMUL_VALUES == ("ieee", "tf32")
    assert DEFAULT_TF32_MATMUL == "ieee"
    assert normalize_tf32_matmul(None) == "ieee"
    assert normalize_tf32_matmul("TF32") == "tf32"
    with pytest.raises(ValueError, match="training.tf32_matmul"):
        normalize_tf32_matmul("auto")
    with pytest.raises(ValueError, match="training.tf32_matmul"):
        normalize_tf32_matmul("high")
    with pytest.raises(ValueError, match="training.tf32_matmul"):
        normalize_tf32_matmul("tensorfloat")


def test_ieee_keeps_flags_off():
    torch_like = _fake_torch(precision="tf32")
    policy = apply_tf32_matmul("ieee", device="cuda:0", torch_module=torch_like)
    assert policy.enabled is False
    assert policy.applied is True
    assert torch_like.backends.cuda.matmul.fp32_precision == "ieee"
    assert torch_like.backends.cudnn.fp32_precision == "ieee"


def test_tf32_resolves_on_ampere_nvidia_cuda():
    torch_like = _fake_torch(cuda=_FakeCuda(available=True, capability=(8, 9)))
    policy = apply_tf32_matmul("tf32", device="cuda:0", torch_module=torch_like)
    assert policy.enabled is True
    assert policy.reason == "nvidia_cuda_tensor_cores"
    assert torch_like.backends.cuda.matmul.fp32_precision == "tf32"
    assert torch_like.backends.cudnn.fp32_precision == "tf32"


def test_tf32_falls_back_to_ieee_on_pre_ampere_cuda():
    torch_like = _fake_torch(cuda=_FakeCuda(available=True, capability=(6, 1)))
    policy = apply_tf32_matmul("tf32", device="cuda:0", torch_module=torch_like)
    assert policy.enabled is False
    assert policy.reason == "cuda_compute_capability_below_sm70"
    assert torch_like.backends.cuda.matmul.fp32_precision == "ieee"


def test_tf32_falls_back_to_ieee_on_rocm_hip_runtime():
    torch_like = _fake_torch(hip="6.2", cuda=_FakeCuda(available=True))
    policy = apply_tf32_matmul("tf32", device="cuda:0", torch_module=torch_like)
    assert policy.enabled is False
    assert policy.reason == "rocm_hip"
    assert torch_like.backends.cuda.matmul.fp32_precision == "ieee"


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_tf32_falls_back_to_ieee_on_non_cuda_devices(device: str):
    torch_like = _fake_torch()
    policy = apply_tf32_matmul("tf32", device=device, torch_module=torch_like)
    assert policy.enabled is False
    assert policy.reason != "nvidia_cuda_tensor_cores"
    assert torch_like.backends.cuda.matmul.fp32_precision == "ieee"


def test_cfg_reader_reads_training_section():
    cfg = OmegaConf.create({"training": {"tf32_matmul": "tf32"}})
    assert resolve_tf32_matmul_from_cfg(cfg).requested == "tf32"
    assert resolve_tf32_matmul_from_cfg({}).requested == "ieee"


def test_apply_from_cfg_flips_flags_for_tf32():
    torch_like = _fake_torch(precision="ieee", cuda=_FakeCuda(available=True))
    cfg = OmegaConf.create({"training": {"tf32_matmul": "tf32"}})
    policy = apply_tf32_matmul_from_cfg(cfg, device="cuda:0", torch_module=torch_like)
    assert policy.enabled is True
    assert torch_like.backends.cuda.matmul.fp32_precision == "tf32"


def test_owner_yaml_surface_is_ppo_and_appo_only_with_ieee_default():
    """#2477: the knob exists only where FP32 matmuls dominate the update."""
    for algo in ("ppo", "appo"):
        text = (_CONF_DIR / algo / "config.yaml").read_text(encoding="utf-8")
        assert "tf32_matmul: ieee" in text, algo
        assert "float32_matmul_precision" not in text, algo
    for algo in ("sac", "flashsac"):
        text = (_CONF_DIR / algo / "config.yaml").read_text(encoding="utf-8")
        assert "tf32_matmul" not in text, algo
        assert "float32_matmul_precision" not in text, algo


def test_runtime_format_helper():
    @dataclass
    class _Policy:
        requested: str = "tf32"
        enabled: bool = True
        reason: str = "nvidia_cuda_tensor_cores"

    assert Tf32Policy is not None
    text = format_tf32_matmul(_Policy())
    assert "tf32" in text
    assert "unspecified" in format_tf32_matmul(None)
    assert resolve_tf32_matmul("ieee").manifest() == {
        "requested": "ieee",
        "enabled": False,
        "reason": "configured_ieee",
        "applied": False,
    }
