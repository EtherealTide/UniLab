"""Training-worker device routing for UniSim backends.

The learner device and the device consumed by a simulator are related, but
they do not always use the same index namespace.  In particular, the
off-policy launcher keeps the host-visible CUDA namespace while the PPO
``torchrun`` launcher remaps ``CUDA_VISIBLE_DEVICES`` and therefore exposes a
rank-local index to child processes.  The helpers in this module keep that
translation on the cold path, next to the backend process binding contract.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

# These backends consume an explicit integer device id while materializing
# their simulator.  MuJoCo/Motrix/Drake run CPU-authoritative physics; their
# optional accelerator Manager buffers are selected separately through
# ``manager_torch_device``.
BACKEND_ENV_DEVICE_FIELDS: dict[str, str] = {
    "isaacgym": "isaacgym_device_id",
    "isaacsim": "isaacsim_device_id",
    "genesis": "genesis_device_id",
}


# Newton consumes an explicit ``cuda:N`` device string (``newton_device``)
# instead of an integer id.  uni_rl's collector-side process binding is
# injection-based: it binds through the caller-supplied ``bind_device``
# callable and fails closed when none is injected.  UniLab injects
# ``bind_backend_process_device_for_backend`` (which covers mjwarp and
# newton); the env override additionally forwards the rank-local device
# string so spawn collectors can pass it to the Newton adapter.
BACKEND_ENV_DEVICE_STR_FIELDS: dict[str, str] = {
    "newton": "newton_device",
}

# Isaac's simulator runs in a dedicated worker, but the host learner owns the
# CUDA IPC arena.  Its Torch current device must therefore agree with the
# integer payload sent to that worker before construction.
_EXTERNAL_CUDA_IPC_BACKENDS = {"isaacgym", "isaacsim"}

# HOST_BRIDGE backends accept optional accelerator Torch carriers across their packed
# boundary. A process may explicitly request those carriers on its learner Torch
# device; unset non-accelerator requests retain the owner/default CPU placement.
_HOST_BRIDGE_TORCH_BACKENDS = {"mujoco", "motrix", "drake", "superdex"}


# Set once ``bind_genesis_process_device`` has pinned CUDA_VISIBLE_DEVICES for
# this process.  Genesis/Quadrants binds its CUDA runtime to the first visible
# device regardless of torch's current device (verified on genesis_world
# 1.3.3 / Quadrants 1.3.0, issue #1508), so a non-zero request is honored by
# shrinking visibility to the target GPU and using the in-process index 0.
# The flag flips the resolution helpers into the pinned namespace.
_genesis_device_pinned = False


def _normalize_backend(backend_type: str) -> str:
    if not isinstance(backend_type, str) or not backend_type.strip():
        raise ValueError(f"backend_type must be a non-empty string, got {backend_type!r}")
    return backend_type.strip().lower()


def _cuda_device_index(device: str | None) -> int | None:
    """Extract an integer CUDA index from a device string.

    ``cuda`` without an explicit suffix is resolved through the current CUDA
    device when possible.  This is deliberately a cold-path helper; it is not
    used from environment ``step``/``reset`` loops.
    """

    if device is None:
        return None
    value = str(device).strip().lower()
    if value == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:
            # Device discovery is only a fallback for an unindexed alias.  A
            # configured topology below remains authoritative if available.
            pass
        return 0
    if not value.startswith("cuda:"):
        return None
    index_text = value.split(":", 1)[1].strip()
    if not index_text:
        raise ValueError(f"CUDA device alias {device!r} has an empty index")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"CUDA device alias {device!r} has a non-integer index") from exc
    if index < 0:
        raise ValueError(f"CUDA device alias {device!r} has a negative index")
    return index


def rank_local_visible_cuda_entries(
    current_visible_devices: str | None = None,
) -> tuple[str, ...]:
    """Return opaque rank-local CUDA visibility entries before Torch init."""
    raw = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if current_visible_devices is None
        else current_visible_devices
    )
    if raw is None:
        return ()
    entries = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    if entries == ("-1",):
        return ()
    return entries


def resolve_backend_env_device_id(
    backend_type: str,
    *,
    learner_device: str | None = None,
) -> int | None:
    """Resolve the simulator payload id from the rank-local CUDA namespace.

    ``None`` means the backend has no payload device field. With CUDA visible,
    exactly one rank-local entry is required and the payload id is always ``0``;
    otherwise the explicit non-Genesis learner device fallback remains valid.
    """

    backend = _normalize_backend(backend_type)
    field = BACKEND_ENV_DEVICE_FIELDS.get(backend) or BACKEND_ENV_DEVICE_STR_FIELDS.get(backend)
    if field is None:
        return None

    visible_entries = rank_local_visible_cuda_entries()
    if len(visible_entries) == 1:
        return 0
    if visible_entries:
        raise ValueError(
            "A CUDA rank must own exactly one CUDA_VISIBLE_DEVICES entry; got "
            f"{','.join(visible_entries)!r}"
        )
    if backend == "genesis" and _genesis_device_pinned:
        return 0
    return _cuda_device_index(learner_device)


def apply_backend_env_device_override(
    env_cfg_override: Mapping[str, Any] | None,
    backend_type: str,
    *,
    learner_device: str | None = None,
) -> dict[str, Any]:
    """Return an env override carrying the rank-selected simulator device.

    The input mapping is never mutated.  If no topology/device can be
    resolved, the owner-configured value is preserved.  This lets one helper
    serve training, playback, and custom entrypoints while retaining the
    historical default (device zero) for single-process calls.
    """

    result = dict(env_cfg_override) if env_cfg_override is not None else {}
    backend = _normalize_backend(backend_type)
    int_field = BACKEND_ENV_DEVICE_FIELDS.get(backend)
    str_field = BACKEND_ENV_DEVICE_STR_FIELDS.get(backend)
    if int_field is None and str_field is None:
        return result
    device_id = resolve_backend_env_device_id(backend, learner_device=learner_device)
    if device_id is None:
        return result
    if str_field is not None:
        result[str_field] = f"cuda:{int(device_id)}"
    elif int_field is not None:
        result[int_field] = int(device_id)
    return result


def apply_manager_torch_device_override(
    env_cfg_override: Mapping[str, Any] | None,
    backend_type: str,
    *,
    learner_device: str | None = None,
) -> dict[str, Any]:
    """Return an env override selecting optional HOST_BRIDGE Torch carriers.

    The input mapping is never mutated. Explicit CPU requests force CPU carriers.
    Unset and non-CUDA learner requests retain owner/default placement. A CUDA request
    is validated against backend capabilities when the environment binds.
    DEVICE_RESIDENT backends own placement and never receive this synthetic field.
    """

    result = dict(env_cfg_override) if env_cfg_override is not None else {}
    if _normalize_backend(backend_type) not in _HOST_BRIDGE_TORCH_BACKENDS:
        return result
    if learner_device is None:
        return result
    device = str(learner_device).strip()
    if not device or device.lower() == "cpu":
        result["manager_torch_device"] = "cpu"
        return result
    if device.split(":", 1)[0].lower() != "cuda":
        return result
    result["manager_torch_device"] = device
    return result


def resolve_backend_process_device(backend_type: str, learner_device: str | None) -> str | None:
    backend = _normalize_backend(backend_type)
    if (
        backend not in {"mjwarp", "newton", "genesis"}
        and backend not in _EXTERNAL_CUDA_IPC_BACKENDS
    ):
        return None
    if learner_device is None:
        raise ValueError(f"{backend} requires an explicit CUDA process device")
    resolved = str(learner_device).strip()
    if resolved.split(":", 1)[0].lower() != "cuda":
        raise ValueError(
            f"{backend} requires a CUDA process device shared with its learner; got {resolved!r}"
        )
    return resolved


def _bind_external_cuda_ipc_process_device(
    backend_type: str,
    resolved: str,
    *,
    backend_device_id: int | None = None,
) -> str:
    import torch

    learner_index = _cuda_device_index(resolved)
    if learner_index is None:
        raise ValueError(
            f"{backend_type} requires a CUDA process device shared with its learner; "
            f"got {resolved!r}"
        )
    if backend_device_id is not None:
        if (
            isinstance(backend_device_id, bool)
            or not isinstance(backend_device_id, int)
            or backend_device_id < 0
        ):
            raise ValueError(
                f"{backend_type} backend device id must be a non-negative integer or None, "
                f"got {backend_device_id!r}"
            )
        if backend_device_id != learner_index:
            raise ValueError(
                f"{backend_type} learner device {resolved!r} does not match its backend "
                f"payload device id {backend_device_id}; refusing to construct cross-device "
                "CUDA IPC"
            )
    if not torch.cuda.is_available():
        raise ValueError(
            f"{backend_type} requires CUDA device {resolved!r}, but CUDA is unavailable "
            "in this process"
        )
    visible_count = int(torch.cuda.device_count())
    if learner_index >= visible_count:
        raise ValueError(
            f"{backend_type} device index {learner_index} is out of range; "
            f"torch.cuda.device_count()={visible_count}"
        )
    torch.cuda.set_device(learner_index)
    return f"cuda:{learner_index}"


def configure_backend_process_device(
    backend_type: str,
    learner_device: str | None,
    *,
    backend_device_id: int | None = None,
) -> str | None:
    resolved = resolve_backend_process_device(backend_type, learner_device)
    if resolved is None:
        return None
    backend = _normalize_backend(backend_type)
    if backend == "genesis":
        return bind_genesis_process_device(resolved)
    if backend in _EXTERNAL_CUDA_IPC_BACKENDS:
        return _bind_external_cuda_ipc_process_device(
            backend,
            resolved,
            backend_device_id=backend_device_id,
        )
    return bind_backend_process_device_for_backend(backend_type, resolved)


def bind_backend_process_device_for_backend(backend_type: str, resolved: str) -> str | None:
    """Bind one backend's process-global accelerator device.

    This top-level callable is intentionally backend-aware and lazy.  It can
    be wrapped with :func:`functools.partial` and injected into uni_rl's
    spawn-based collectors while remaining pickleable by module reference.
    """
    backend = _normalize_backend(backend_type)
    if backend == "newton":
        from unisim.backend.newton.runtime import bind_newton_process_device

        return cast(str | None, bind_newton_process_device(resolved))
    if backend == "mjwarp":
        from unisim.backend.mjwarp.runtime import bind_mjwarp_process_device

        return cast(str | None, bind_mjwarp_process_device(resolved))
    return None


def bind_backend_process_device(resolved: str) -> str | None:
    """Bind a resolved backend process device in the current process.

    Top-level on purpose: uni_rl's off-policy collectors receive this as the
    injected ``backend_device_binder`` and pickle it by reference into
    spawn-based subprocesses. The mjwarp import stays lazy so the binder is
    importable without the ``mjwarp`` extra installed.
    """
    return bind_backend_process_device_for_backend("mjwarp", resolved)


def _pin_cuda_visible_devices(index: int) -> None:
    """Shrink ``CUDA_VISIBLE_DEVICES`` to the single entry at ``index``.

    The index addresses the *current* visibility namespace: with no variable
    set it is the host index, otherwise it indexes into the existing entries
    (which may be physical indices or UUIDs).  This only works before the
    first CUDA context exists, so an already-initialized torch runtime fails
    closed with an actionable error instead of crashing inside the engine.
    """

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "genesis device routing must pin CUDA_VISIBLE_DEVICES before any CUDA "
            "context is created in this process, but torch CUDA is already "
            "initialized; move configure_backend_process_device earlier in the "
            "entrypoint (before seeding/learner construction)"
        )
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()] if raw else None
    if entries is None:
        count = int(torch.cuda.device_count())
        if index >= count:
            raise ValueError(
                f"genesis device index {index} is out of range; torch.cuda.device_count()={count}"
            )
        target = str(index)
    else:
        if index >= len(entries):
            raise ValueError(
                f"genesis device index {index} is out of range for "
                f"CUDA_VISIBLE_DEVICES={raw!r} ({len(entries)} entr(ies))"
            )
        target = entries[index]
    os.environ["CUDA_VISIBLE_DEVICES"] = target
    # ``device_count`` is lru-cached; drop any pre-pin host-wide count so
    # later callers observe the pinned single-device namespace.
    cache_clear = getattr(torch.cuda.device_count, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def pin_genesis_device_before_cuda_init(
    backend_type: str,
    *,
    learner_device: str | None = None,
) -> str | None:
    """Pin Genesis to its rank device before the first torch CUDA call.

    ``torch.cuda.is_available()`` already latches ``CUDA_VISIBLE_DEVICES`` in
    the CUDA runtime, so the pin must run ahead of *any* torch CUDA query —
    entrypoints should call this before registry/bootstrap/device detection.
    An explicit non-Genesis ``cuda:N`` learner device resolves without touching torch.  Returns the
    in-process device the caller must use when a pin happened, else ``None``.
    """

    if _normalize_backend(backend_type) != "genesis":
        return None
    device_id = resolve_backend_env_device_id(backend_type, learner_device=learner_device)
    if not device_id:
        return None
    return bind_genesis_process_device(f"cuda:{device_id}")


def bind_genesis_process_device(resolved: str) -> str:
    """Select the CUDA device used by an in-process Genesis worker.

    Genesis initializes a process-wide session whose Quadrants CUDA runtime
    always binds the first entry of ``CUDA_VISIBLE_DEVICES``; torch's current
    device alone is *not* honored (issue #1508).  A non-zero request is
    therefore honored by pinning visibility to the target GPU before any CUDA
    context exists, after which the in-process device is ``cuda:0``.  The
    returned string is the device the rest of this process must actually use —
    callers that computed a pre-pin device (learner, probes, collectors) have
    to adopt the returned value.  Binding must happen before constructing the
    backend (and before ``gs.init``), including in spawn-based collector
    processes, and remains in effect for the lifetime of the process.
    """

    global _genesis_device_pinned
    device = str(resolved).strip()
    index = _cuda_device_index(device)
    if index is None:
        raise ValueError(f"genesis requires a CUDA process device; got {resolved!r}")
    import torch

    # Pin *before* any torch CUDA query: even ``torch.cuda.is_available()``
    # latches CUDA_VISIBLE_DEVICES in the runtime, after which rewriting it
    # would silently keep the process on the first previously visible GPU.
    if index > 0:
        if not _genesis_device_pinned:
            _pin_cuda_visible_devices(index)
            _genesis_device_pinned = True
        # Already pinned (or just pinned): the only valid in-process device is
        # index 0.  A stale pre-pin index from the same rank maps onto it.
        index = 0
    if not torch.cuda.is_available():
        raise ValueError(
            f"genesis requires CUDA device {device!r}, but CUDA is unavailable in this process"
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def _reset_genesis_device_pin_for_tests() -> None:
    """Clear the process pin latch; test-only seam (the CVD rewrite itself is
    reverted via ``monkeypatch.setitem``/``delitem`` on ``os.environ``)."""

    global _genesis_device_pinned
    _genesis_device_pinned = False


__all__ = [
    "BACKEND_ENV_DEVICE_FIELDS",
    "apply_backend_env_device_override",
    "apply_manager_torch_device_override",
    "bind_backend_process_device",
    "bind_backend_process_device_for_backend",
    "bind_genesis_process_device",
    "configure_backend_process_device",
    "pin_genesis_device_before_cuda_init",
    "rank_local_visible_cuda_entries",
    "resolve_backend_env_device_id",
    "resolve_backend_process_device",
]
