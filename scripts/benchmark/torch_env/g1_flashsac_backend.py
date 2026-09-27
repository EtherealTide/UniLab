#!/usr/bin/env python3
"""Phase-local G1 FlashSAC backend tensor benchmark.

This minimal probe combines the real G1 owner scene/backend with the existing
Torch motion-tracking manager kernel. It is not an end-to-end learner/run
benchmark: inference IPC, replay insertion, and the production Manager-Based
dispatch are intentionally excluded. The measured phases are physics/backend
exchange, Update State, reset-row selection, Reset Done, backend reset, and
reset-state publication. Each phase ends at a Torch synchronization point, so
throughput is reported for environment control steps (three physics substeps).
Run it inside the sibling checkout workspace: this branch resolves UniSim,
UniLab-RL, and mjbatch through the relative paths in ``pyproject.toml``.
Each requested backend is measured in its own Python process to avoid sharing
 warmed vendor context and allocator state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.benchmark.torch_env.motion_tracking import (  # noqa: E402
    CLIP_END_FRAME,
    MotionTrackingWorkload,
)
from scripts.benchmark.torch_env.xp import TorchBackend, TorchRng  # noqa: E402


def _build_cfg(backend: str, num_envs: int) -> Any:
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg

    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"),
        version_base="1.3",
    ):
        owner_cfg = compose(
            config_name="config",
            overrides=[
                f"task=g1_motion_tracking/{backend}",
                f"algo.num_envs={num_envs}",
                "training.no_play=true",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    override = BackendAdapter(
        owner_cfg, root_dir=ROOT_DIR, algo_name="flashsac"
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def _build_backend(backend: str, num_envs: int) -> Any:
    from unilab.base.backend_factory import create_backend, env_backend_kwargs

    cfg = _build_cfg(backend, num_envs)
    cfg.validate()
    assert cfg.scene is not None
    robot = cfg.scene.entities["robot"]
    kwargs = env_backend_kwargs(cfg)
    kwargs["base_name"] = robot.root_body_name
    if backend == "mujoco":
        kwargs["tracked_body_names"] = tuple(robot.body_names)
    return create_backend(
        backend,
        cfg.scene,
        num_envs,
        cfg.sim_dt,
        body_state_required=True,
        **kwargs,
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_source_info(path: Path) -> dict[str, str | bool | None]:
    root = next(
        (candidate for candidate in (path, *path.parents) if (candidate / ".git").exists()),
        None,
    )
    if root is None:
        return {"git_root": None, "branch": None, "commit": None, "dirty": None}

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def _worktree_digest() -> str | None:
        patch = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=False,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            check=False,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256()
        digest.update(b"git-diff-binary\0")
        digest.update(patch)
        digest.update(b"untracked-files\0")
        for relative in untracked.split(b"\0"):
            if not relative:
                continue
            source = root / relative.decode(errors="surrogateescape")
            if not source.is_file():
                continue
            digest.update(relative + b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    return {
        "git_root": str(root),
        "branch": git("branch", "--show-current") or None,
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(git("status", "--porcelain")),
        "worktree_patch_sha256": _worktree_digest() if root is not None else None,
    }


def _package_source_info(name: str) -> dict[str, str | bool | None]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return {"path": None, **_git_source_info(Path.cwd())}
    path = Path(spec.origin).resolve()
    return {"path": str(path), **_git_source_info(path)}


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _nvidia_smi_fields() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.sm,clocks.mem,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    names = ("driver_version", "sm_clock_mhz", "memory_clock_mhz", "power_draw_w", "power_limit_w")
    values = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not values:
        return {}
    return {
        name: value.strip().removesuffix(" W").removesuffix(" MHz")
        for name, value in zip(names, values.split(","), strict=False)
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "iters": len(values),
    }


def _reset_row_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "iters": len(values),
    }


def _backend_timing_stats(results: list[dict[str, Any] | None]) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = {}
    for result in results:
        for key, value in (result or {}).get("timing", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.setdefault(key, []).append(float(value))
    return {key: _stats(values) for key, values in samples.items()}


def _run(backend: str, num_envs: int, warmup: int, iters: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("G1 FlashSAC tensor benchmark requires CUDA")
    device = torch.device("cuda", index=torch.cuda.current_device())
    robot_body_names = tuple(_build_cfg(backend, num_envs).scene.entities["robot"].body_names)
    sim_backend = _build_backend(backend, num_envs)
    xp = TorchBackend("cuda")
    workload = MotionTrackingWorkload(
        xp, TorchRng("cuda"), seed=7, vectorized_reset_rng=True, num_envs=num_envs
    )
    mode = sim_backend.tensor_execution().value
    if mode not in {"device_resident", "host_bridge"}:
        raise RuntimeError(f"backend {backend} did not negotiate a tensor lifecycle: {mode}")
    sim_backend.materialize()
    sim_backend.reset()
    robot_body_ids_host = sim_backend.get_body_ids(tuple(robot_body_names))

    action = torch.zeros((num_envs, sim_backend.num_actuators), device=device)
    reset_rows = max(1, num_envs // 16)
    row_pattern = torch.arange(num_envs, device=device) % 16
    device_state_views = sim_backend.get_state_views(("qpos", "qvel"), device=device)
    device_sensor_views = (
        {
            "linvel": sim_backend.get_sensor_view("pelvis_local_linvel", device=device),
            "gyro": sim_backend.get_sensor_view("torso_gyro", device=device),
            "body_pos": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_pos_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_quat": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_quat_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_lin_vel": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_linvel_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
            "body_ang_vel": torch.stack(
                tuple(
                    sim_backend.get_sensor_view(f"track_angvel_w_{name}", device=device)
                    for name in robot_body_names
                ),
                dim=1,
            ),
        }
        if mode == "device_resident"
        else None
    )
    phase_samples = {
        "backend_step_ms": [],
        "state_exchange_ms": [],
        "update_state_ms": [],
        "reset_selection_ms": [],
        "reset_done_ms": [],
        "backend_reset_ms": [],
        "reset_publish_ms": [],
        "iteration_ms": [],
    }
    reset_row_samples: list[float] = []
    backend_step_results: list[dict[str, Any] | None] = []
    backend_reset_results: list[dict[str, Any] | None] = []

    # Keep the synthetic action/reset schedule identical across backends even
    # when multiple runs share one CUDA context.
    torch.cuda.manual_seed(7)

    try:
        for iteration in range(warmup + iters):
            if iteration == warmup:
                torch.cuda.synchronize()
                phase_samples = {name: [] for name in phase_samples}
                reset_row_samples = []
                backend_step_results = []
                backend_reset_results = []

            action.uniform_(-1.0, 1.0)
            workload.current_actions.copy_(action)
            started = time.perf_counter()

            phase = time.perf_counter()
            backend_step_results.append(sim_backend.step_tensor(action, nsteps=3))
            phase_samples["backend_step_ms"].append((time.perf_counter() - phase) * 1000.0)

            phase = time.perf_counter()
            if mode == "device_resident":
                workload.dof_pos = device_state_views["qpos"][:, 7:]
                workload.dof_vel = device_state_views["qvel"][:, 6:]
                assert device_sensor_views is not None
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_pos_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=device_sensor_views["body_pos"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_quat_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=device_sensor_views["body_quat"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_linvel_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=device_sensor_views["body_lin_vel"],
                )
                torch.stack(
                    tuple(
                        sim_backend.get_sensor_view(f"track_angvel_w_{name}", device=device)
                        for name in robot_body_names
                    ),
                    dim=1,
                    out=device_sensor_views["body_ang_vel"],
                )
                workload.body_pos = device_sensor_views["body_pos"]
                workload.body_quat = device_sensor_views["body_quat"]
                workload.body_lin_vel = device_sensor_views["body_lin_vel"]
                workload.body_ang_vel = device_sensor_views["body_ang_vel"]
                workload.linvel = device_sensor_views["linvel"]
                workload.gyro = device_sensor_views["gyro"]
            else:
                state = sim_backend.get_state_views(("qpos", "qvel"), device=device)
                workload.dof_pos = state["qpos"][:, 7:]
                workload.dof_vel = state["qvel"][:, 6:]
                workload.body_pos = torch.from_numpy(
                    np.ascontiguousarray(sim_backend.get_body_pos_w(robot_body_ids_host))
                ).to(device)
                workload.body_quat = torch.from_numpy(
                    np.ascontiguousarray(sim_backend.get_body_quat_w(robot_body_ids_host))
                ).to(device)
                workload.body_lin_vel = torch.from_numpy(
                    np.ascontiguousarray(sim_backend.get_body_lin_vel_w(robot_body_ids_host))
                ).to(device)
                workload.body_ang_vel = torch.from_numpy(
                    np.ascontiguousarray(sim_backend.get_body_ang_vel_w(robot_body_ids_host))
                ).to(device)
                workload.linvel = sim_backend.get_sensor_view("pelvis_local_linvel", device=device)
                workload.gyro = sim_backend.get_sensor_view("torso_gyro", device=device)
            xp.sync()
            phase_samples["state_exchange_ms"].append((time.perf_counter() - phase) * 1000.0)

            phase = time.perf_counter()
            obs, reward, terminated = workload.update_state(should_log=False)
            xp.sync()
            phase_samples["update_state_ms"].append((time.perf_counter() - phase) * 1000.0)
            del obs, reward

            # Keep a comparable scheduled-reset floor while resetting clip-end
            # rows immediately so subsequent frame gathers remain in range.
            phase = time.perf_counter()
            mask = (row_pattern == (iteration % 16)) | (workload.current_frames > CLIP_END_FRAME)
            env_ids = mask.nonzero(as_tuple=False).reshape(-1)
            xp.sync()
            phase_samples["reset_selection_ms"].append((time.perf_counter() - phase) * 1000.0)
            reset_row_samples.append(float(env_ids.shape[0]))
            phase = time.perf_counter()
            qpos, qvel, reset_obs, _ = workload.reset_done(env_ids)
            xp.sync()
            phase_samples["reset_done_ms"].append((time.perf_counter() - phase) * 1000.0)
            del reset_obs

            phase = time.perf_counter()
            backend_reset_results.append(sim_backend.set_state_tensor(env_ids, qpos, qvel))
            phase_samples["backend_reset_ms"].append((time.perf_counter() - phase) * 1000.0)
            phase = time.perf_counter()
            workload.dof_pos[env_ids] = qpos[:, 7:]
            workload.dof_vel[env_ids] = qvel[:, 6:]
            xp.sync()
            phase_samples["reset_publish_ms"].append((time.perf_counter() - phase) * 1000.0)
            phase_samples["iteration_ms"].append((time.perf_counter() - started) * 1000.0)
    finally:
        sim_backend.close()

    stats = {name: _stats(values) for name, values in phase_samples.items()}
    mean_total_s = stats["iteration_ms"]["mean_ms"] / 1000.0
    return {
        "backend": backend,
        "tensor_execution": mode,
        "num_envs": num_envs,
        "physics_substeps_per_control_step": 3,
        "warmup": warmup,
        "iters": iters,
        "scheduled_reset_rows": reset_rows,
        "reset_rows": _reset_row_stats(reset_row_samples),
        "reset_policy": {
            "scheduled_cadence_control_steps": 16,
            "scheduled_rows_target": reset_rows,
            "clip_end_overflow_reset": True,
            "terminated_reset": False,
        },
        "seeds": {
            "numpy_synthetic_workload": 7,
            "torch_cuda": 7,
        },
        "throughput_env_control_steps_per_s": num_envs / mean_total_s,
        "throughput_physics_substeps_per_s": 3 * num_envs / mean_total_s,
        "phases": stats,
        "backend_timings": {
            "step_tensor": _backend_timing_stats(backend_step_results),
            "set_state_tensor": _backend_timing_stats(backend_reset_results),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="mjwarp,mujoco")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_single-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_quiet", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._single_result is not None:
        result = _run(
            args.backends.removesuffix(",").removeprefix(",").strip(),
            args.num_envs,
            args.warmup,
            args.iters,
        )
        args._single_result.parent.mkdir(parents=True, exist_ok=True)
        args._single_result.write_text(json.dumps({"result": result}) + "\n")
        return

    results = []
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    with tempfile.TemporaryDirectory(prefix="g1-flashsac-tensor-") as temporary_dir:
        for index, backend in enumerate(backends):
            print(f"benchmarking {backend} in an isolated process...", flush=True)
            result_path = Path(temporary_dir) / f"result-{index}.json"
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--backends",
                    backend,
                    "--num-envs",
                    str(args.num_envs),
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--_single-result",
                    str(result_path),
                    "--_quiet",
                ],
                check=True,
                cwd=ROOT_DIR,
            )
            result = json.loads(result_path.read_text())["result"]
            results.append(result)
            print(
                f"{backend}: {result['throughput_env_control_steps_per_s']:,.0f} "
                "env control steps/s; "
                f"iteration={result['phases']['iteration_ms']['mean_ms']:.3f} ms",
                flush=True,
            )
    cuda_index = torch.cuda.current_device()
    payload = {
        "schema_version": "0.1.0",
        "scope": "phase-local",
        "excluded_components": [
            "inference_ipc",
            "replay_ingestion",
            "learner_update",
            "production_manager_dispatch",
        ],
        "timing_semantics": "synchronize_torch_after_each_phase",
        "process_isolation": "one_process_per_backend",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "versions": {
            "installed_unisim_core_metadata": _package_version("unisim-core"),
            "mujoco_warp": _package_version("mujoco_warp"),
            "mujoco": _package_version("mujoco"),
            "warp_lang": _package_version("warp-lang"),
            "mjbatch_uni": _package_version("mjbatch-uni"),
        },
        "unisim_source": _package_source_info("unisim"),
        "local_dependencies": {
            "unilab_rl": _package_source_info("uni_rl"),
            "mjbatch_uni": _git_source_info(ROOT_DIR.parent / "mjbatch_uni"),
        },
        "runtime": {
            "torch_cuda": torch.version.cuda,
            "torch_cudnn": torch.backends.cudnn.version(),
            "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(cuda_index))),
            "gpu_total_memory_gib": (
                torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3
            ),
            **_nvidia_smi_fields(),
        },
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
        },
        "cuda_device": torch.cuda.get_device_name(cuda_index),
        "cuda_device_index": cuda_index,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not args._quiet:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
