"""Torch device aliases and shared device-runtime primitives for RL.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.utils.device``. ``get_default_device`` is owned by
:mod:`unilab.utils.device` and re-exported here (module path, per repo rules).
Backend process-device resolution/binding is not owned by this layer: runners
and workers receive it through injected callables (see
:mod:`unilab.rl.runtime.offpolicy.double_buffer_runner` and
:mod:`unilab.rl.runtime.offpolicy.worker`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from unilab.utils.device import get_default_device, mps_available, xpu_available

__all__ = [
    "CudaDeviceTimer",
    "DeviceTimer",
    "SynchronizedDeviceTimer",
    "device_runtime_capabilities",
    "get_default_device",
    "make_device_timer",
    "resolve_torch_device_alias",
    "sync_device",
]


def sync_device(device: str | torch.device, *, stream: bool = False) -> None:
    """Wait for device work using the platform's public synchronization API.

    ``stream=True`` waits on the device's current stream.  It is CUDA-specific:
    MPS exposes no stream-scoped synchronization API, so requesting it there
    fails closed rather than silently draining unrelated queued work.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if stream:
            torch.cuda.current_stream(resolved).synchronize()
        else:
            torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        if stream:
            raise ValueError("torch.mps does not expose stream synchronization")
        torch.mps.synchronize()
    elif resolved.type != "cpu":
        raise ValueError(f"Unsupported device synchronization type: {resolved.type!r}")


class DeviceTimer(Protocol):
    """Portable wall-time-compatible timing boundary for accelerator work."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def elapsed_s(self) -> float: ...


class SynchronizedDeviceTimer:
    """Wall-clock timer whose elapsed read owns the device synchronization.

    ``stop`` only records wall time, allowing action publication or another
    host boundary to occur first.  ``elapsed_s`` synchronizes before returning.
    """

    def __init__(self, device: str | torch.device) -> None:
        self._device = torch.device(device)

    def start(self) -> None:
        self._start_ns = time.perf_counter_ns()

    def stop(self) -> None:
        self._end_ns = time.perf_counter_ns()

    def elapsed_s(self) -> float:
        sync_device(self._device)
        return (self._end_ns - self._start_ns) / 1e9


class CudaDeviceTimer:
    """CUDA event timer whose elapsed read owns the synchronization point."""

    def __init__(self, device: str | torch.device) -> None:
        self._device = torch.device(device)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start_event.record(torch.cuda.current_stream(self._device))

    def stop(self) -> None:
        self._end_event.record(torch.cuda.current_stream(self._device))

    def elapsed_s(self) -> float:
        self._end_event.synchronize()
        return float(self._start_event.elapsed_time(self._end_event) / 1e3)


def make_device_timer(device: str | torch.device) -> DeviceTimer:
    """Build the validated timing implementation for ``device``."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        return CudaDeviceTimer(device)
    if device_type in {"mps", "cpu"}:
        return SynchronizedDeviceTimer(device)
    raise ValueError(f"Unsupported device timer type: {device_type!r}")


@dataclass(frozen=True)
class DeviceRuntimeCapabilities:
    """Explicit capability/policy labels recorded in runtime diagnostics."""

    device_type: str
    sync_api: str
    stream_sync_api: str | None
    has_timing_events: bool

    def manifest(self) -> dict[str, str | bool | None]:
        return {
            "device_type": self.device_type,
            "sync_api": self.sync_api,
            "stream_sync_api": self.stream_sync_api,
            "has_timing_events": self.has_timing_events,
            "sync_policy": "capability",
            "timing_policy": "capability",
        }


def device_runtime_capabilities(device: str | torch.device) -> DeviceRuntimeCapabilities:
    """Return the supported public synchronization/timing surface."""
    device_type = torch.device(device).type
    if device_type == "cuda":
        return DeviceRuntimeCapabilities(
            device_type=device_type,
            sync_api="torch.cuda.synchronize",
            stream_sync_api="torch.cuda.current_stream.synchronize",
            has_timing_events=True,
        )
    if device_type == "mps":
        return DeviceRuntimeCapabilities(
            device_type=device_type,
            sync_api="torch.mps.synchronize",
            stream_sync_api=None,
            has_timing_events=False,
        )
    if device_type != "cpu":
        raise ValueError(f"Unsupported device capability type: {device_type!r}")
    return DeviceRuntimeCapabilities(
        device_type=device_type,
        sync_api="none",
        stream_sync_api=None,
        has_timing_events=False,
    )


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
