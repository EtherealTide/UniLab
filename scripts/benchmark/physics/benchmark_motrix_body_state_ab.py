"""Local A/B microbenchmark for Motrix selected-body world-state reads (#1308).

Measures the UniLab motion-tracking body-state read pattern (8192 envs x 14
tracked bodies) across four paths against the same ``MotrixBackend`` instance:

* ``legacy_copy`` — the pre-fused path: ``SimBackend.copy_body_state_w``
  default, i.e. cache-based ``get_body_state_w`` (full-link pose/velocity
  materialization + selected gather) followed by a copy into caller buffers.
* ``fused_packed`` — one native ``SceneModel.get_link_states`` call for the
  selected link ids returning packed arrays, then assigned into caller
  buffers (no caller-owned native outs).
* ``fused_out`` — the shipped ``MotrixBackend.copy_body_state_w``: one fused
  native read writing directly into caller-owned buffers.
* ``getter_cache`` / ``getter_fused`` — allocating ``get_body_state_w``
  variants (cache-based vs fused packed) to decide whether the allocating
  getter should also be rerouted.

By default the link-velocity cache is invalidated before every timed call to
mimic the per-step hot path, where the fused copy is the only velocity
consumer and the cache is cold each step. ``--warm-cache`` disables that.

Usage::

    uv run scripts/benchmark/physics/benchmark_motrix_body_state_ab.py \
        --num-envs 8192 --iters 50 --warmup 10
    uv run scripts/benchmark/physics/benchmark_motrix_body_state_ab.py \
        --out scripts/benchmark/outputs/body_state_ab.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unisim.backend.base import SimBackend  # noqa: E402

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.scene import SceneCfg  # noqa: E402

_G1_MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")

# Tracked bodies of sac/g1_motion_tracking (conf/sac/task/g1_motion_tracking/base.yaml).
_TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

_COPY_ARMS = ("legacy_copy", "fused_packed", "fused_out")
_GETTER_ARMS = ("getter_cache", "getter_fused")


def _make_backend(num_envs: int):
    from unisim.backend.motrix.backend import MotrixBackend

    from unilab.assets.hub import ensure_robot_assets_for_paths

    ensure_robot_assets_for_paths([_G1_MODEL_FILE])
    return MotrixBackend(
        SceneCfg(model_file=_G1_MODEL_FILE),
        num_envs,
        sim_dt=0.005,
        base_name="pelvis",
    )


def _legacy_copy(backend, ids: np.ndarray, outs: tuple[np.ndarray, ...]) -> None:
    # Unbound base-class default: cache-based get_body_state_w + copy.
    SimBackend.copy_body_state_w(backend, ids, *outs)


def _fused_packed(backend, ids: np.ndarray, outs: tuple[np.ndarray, ...]) -> None:
    pos, quat, lin_vel, ang_vel = backend._model.get_link_states(
        backend._data, indices=ids.tolist(), quat_order="wxyz"
    )
    outs[0][...] = pos
    outs[1][...] = quat
    outs[2][...] = lin_vel
    outs[3][...] = ang_vel


def _fused_out(backend, ids: np.ndarray, outs: tuple[np.ndarray, ...]) -> None:
    backend.copy_body_state_w(ids, *outs)


def _getter_cache(backend, ids: np.ndarray) -> tuple[np.ndarray, ...]:
    return backend.get_body_state_w(ids)


def _getter_fused(backend, ids: np.ndarray) -> tuple[np.ndarray, ...]:
    return backend._model.get_link_states(backend._data, indices=ids.tolist(), quat_order="wxyz")


def _check_parity(
    backend,
    ids: np.ndarray,
    dtype: np.dtype,
) -> None:
    """All arms must agree with the legacy path within the test tolerances."""
    num_envs, num_bodies = backend.num_envs, len(ids)
    ref_outs = _alloc_outs(num_envs, num_bodies, dtype)
    _legacy_copy(backend, ids, ref_outs)

    arm_outs = _alloc_outs(num_envs, num_bodies, dtype)
    for name, fn in (("fused_packed", _fused_packed), ("fused_out", _fused_out)):
        fn(backend, ids, arm_outs)
        for ref, got, label in zip(ref_outs, arm_outs, ("pos", "quat", "lin_vel", "ang_vel")):
            tol = 1e-5 if label in ("pos", "quat") else 1e-4
            if not np.allclose(ref, got, atol=tol, rtol=tol):
                raise AssertionError(f"{name} {label} diverges from legacy path")

    for name, fn in (("getter_cache", _getter_cache), ("getter_fused", _getter_fused)):
        got = fn(backend, ids)
        for ref, arr, label in zip(ref_outs, got, ("pos", "quat", "lin_vel", "ang_vel")):
            tol = 1e-5 if label in ("pos", "quat") else 1e-4
            if not np.allclose(ref, arr, atol=tol, rtol=tol):
                raise AssertionError(f"{name} {label} diverges from legacy path")


def _alloc_outs(num_envs: int, num_bodies: int, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    return (
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
        np.zeros((num_envs, num_bodies, 4), dtype=dtype),
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
        np.zeros((num_envs, num_bodies, 3), dtype=dtype),
    )


def _time_arm(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    invalidate_velocity_cache: Callable[[], None] | None,
) -> dict[str, float]:
    samples: list[float] = []
    for step in range(warmup + iters):
        if invalidate_velocity_cache is not None:
            invalidate_velocity_cache()
        t0 = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if step >= warmup:
            samples.append(elapsed_ms)
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _format_table(results: dict[str, dict[str, float]], baseline: str) -> str:
    header = f"{'arm':16s} {'mean ms':>10s} {'median ms':>10s} {'min ms':>10s} {'max ms':>10s} {'speedup':>9s}"
    rows = [header, "-" * len(header)]
    base = results[baseline]["median"]
    for name, stats in results.items():
        speedup = base / stats["median"] if stats["median"] > 1e-9 else float("inf")
        rows.append(
            f"{name:16s} {stats['mean']:10.4f} {stats['median']:10.4f}"
            f" {stats['min']:10.4f} {stats['max']:10.4f} {speedup:8.2f}x"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Do not invalidate the link-velocity cache before each timed call.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    backend = _make_backend(args.num_envs)
    assert not backend._portable_mode, "A/B benchmark targets the whole-MJCF path"
    print(f"MotrixBackend ready: num_envs={backend.num_envs}, dt=0.005")

    ids = backend.get_body_ids(_TRACKED_BODIES)
    # Populate state so pose/velocity reads see stepped data.
    backend.step(np.zeros((args.num_envs, backend.num_actuators), dtype=np.float32))

    dtype = backend.get_body_pos_w(ids).dtype
    outs = _alloc_outs(args.num_envs, len(ids), dtype)

    invalidate = None if args.warm_cache else backend._invalidate_link_velocity_cache
    _check_parity(backend, ids, dtype)
    print("parity: all arms agree with the legacy path")

    copy_fns: dict[str, Callable[[], None]] = {
        "legacy_copy": lambda: _legacy_copy(backend, ids, outs),
        "fused_packed": lambda: _fused_packed(backend, ids, outs),
        "fused_out": lambda: _fused_out(backend, ids, outs),
    }
    getter_fns: dict[str, Callable[[], None]] = {
        "getter_cache": lambda: _getter_cache(backend, ids),
        "getter_fused": lambda: _getter_fused(backend, ids),
    }

    # Round-robin across arms per iteration block to cancel state drift: each
    # arm is timed independently but shares the same backend/session.
    results: dict[str, Any] = {"copy": {}, "getter": {}}
    for name, fn in copy_fns.items():
        results["copy"][name] = _time_arm(
            fn, warmup=args.warmup, iters=args.iters, invalidate_velocity_cache=invalidate
        )
    for name, fn in getter_fns.items():
        results["getter"][name] = _time_arm(
            fn, warmup=args.warmup, iters=args.iters, invalidate_velocity_cache=invalidate
        )

    print("\n== copy into caller-owned buffers (8192x14 pattern) ==")
    print(_format_table(results["copy"], "legacy_copy"))
    print("\n== allocating get_body_state_w variants ==")
    print(_format_table(results["getter"], "getter_cache"))

    payload = {
        "num_envs": args.num_envs,
        "num_bodies": len(ids),
        "dtype": str(dtype),
        "iters": args.iters,
        "warmup": args.warmup,
        "warm_cache": args.warm_cache,
        "results": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
