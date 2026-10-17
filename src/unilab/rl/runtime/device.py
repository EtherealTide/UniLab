"""Torch device alias resolution for the RL runtime.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.device``. ``get_default_device`` is owned by
:mod:`unilab.utils.device` and re-exported here (module path, per repo rules).
Backend process-device resolution/binding is not owned by this layer: runners
and workers receive it through injected callables (see
:mod:`unilab.rl.runtime.offpolicy.double_buffer_runner` and
:mod:`unilab.rl.runtime.offpolicy.worker`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from unilab.utils.device import get_default_device, mps_available, xpu_available

__all__ = ["get_default_device", "resolve_torch_device_alias"]


def _device_count(device_type: str) -> int | None:
    if device_type == "cuda":
        return int(torch.cuda.device_count())
    if device_type == "xpu":
        xpu = getattr(torch, "xpu", None)
        device_count = getattr(xpu, "device_count", None)
        if callable(device_count):
            return int(cast(Callable[[], int], device_count)())
    return None


def _parse_device_alias(value: str) -> tuple[str, int | None]:
    raw = value.strip().lower()
    if not raw:
        raise ValueError("Device alias must not be empty")
    if ":" not in raw:
        return raw, None
    base, index_text = raw.split(":", 1)
    if not index_text:
        raise ValueError(f"Device alias {value!r} has an empty index")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Device alias {value!r} has a non-integer index") from exc
    if index < 0:
        raise ValueError(f"Device alias {value!r} has a negative index")
    return base, index


def _resolve_indexed_device(device_type: str, index: int | None, original: str) -> str:
    count = _device_count(device_type)
    if count is not None and index is not None and index >= count:
        raise ValueError(
            f"Requested device {original!r} resolves to {device_type}:{index}, "
            f"but only {count} {device_type} device(s) are available"
        )
    return device_type if index is None else f"{device_type}:{index}"


def _resolve_mps_alias(index: int | None, original: str) -> str:
    if not mps_available():
        raise ValueError(f"Requested device {original!r} requires MPS, but MPS is unavailable")
    if index not in (None, 0):
        raise ValueError(
            f"Requested device {original!r} cannot be mapped to MPS; only index 0 is valid"
        )
    return "mps"


def resolve_torch_device_alias(device: str | None, *, default: str = "cpu") -> str:
    """Resolve a cross-platform torch device alias to a concrete device string.

    ``gpu`` is an abstract accelerator alias and may resolve to an available
    accelerator family. ``cuda`` is a concrete device family and fails closed
    when CUDA is unavailable.
    The function validates the resolved device and never silently falls back to
    CPU for unavailable accelerators.
    """
    original = default if device is None else str(device)
    base, index = _parse_device_alias(original)

    if base == "cpu":
        if index is not None:
            raise ValueError(f"CPU device {original!r} must not include an index")
        return "cpu"

    if base == "mps":
        return _resolve_mps_alias(index, original)

    if base == "xpu":
        if not xpu_available():
            raise ValueError(f"Requested device {original!r} requires XPU, but XPU is unavailable")
        return _resolve_indexed_device("xpu", index, original)

    if base == "cuda":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        raise ValueError(f"Requested device {original!r} requires CUDA, but CUDA is unavailable")

    if base == "gpu":
        if torch.cuda.is_available():
            return _resolve_indexed_device("cuda", index, original)
        if xpu_available():
            return _resolve_indexed_device("xpu", index, original)
        if mps_available():
            return _resolve_mps_alias(index, original)
        raise ValueError(
            f"Requested device {original!r} requires an accelerator, but none is available"
        )

    raise ValueError(f"Unsupported device alias {original!r}; expected cpu, gpu, cuda, mps, or xpu")
