"""Resource and progress monitoring for long-duration training soaks.

The monitor deliberately uses ``/proc`` and ``nvidia-smi`` instead of importing
Torch: it must remain usable when the supervised trainer, its collector
subprocesses, and their accelerator contexts are the only CUDA owners.  The
artifact is intended to be self-contained enough to diagnose a failed soak even
after all child processes have exited.
"""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from unilab.cli import build_command

_ARTIFACT_SCHEMA_VERSION = "0.1.0"


class SoakFailureError(RuntimeError):
    """Raised when a supervised soak violates a fail-closed condition."""


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _proc_dir(pid: int) -> Path:
    return Path("/proc") / str(pid)


def _process_start_ticks(pid: int) -> set[int]:
    """Return Linux thread start ticks, used to reject PID reuse."""
    starts: set[int] = set()
    task_dir = _proc_dir(pid) / "task"
    try:
        entries = list(task_dir.iterdir())
    except OSError:
        return starts
    for task in entries:
        try:
            stat = (task / "stat").read_text(encoding="utf-8")
            fields = stat.rsplit(")", 1)[-1].split()
            # Field 22 (`starttime`) is index 19 after removing pid/comm.
            if len(fields) > 19:
                starts.add(int(fields[19]))
        except (OSError, ValueError):
            continue
    return starts


def _discover_descendants(pid: int) -> dict[int, set[int]]:
    """Discover a process tree once without relying on ``ps`` output."""
    discovered: dict[int, set[int]] = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        if current in discovered:
            continue
        discovered[current] = _process_start_ticks(current)
        children: set[int] = set()
        try:
            task_paths = list((_proc_dir(current) / "task").glob("*"))
            for task in task_paths:
                child_text = (task / "children").read_text(encoding="utf-8")
                children.update(int(value) for value in child_text.split())
        except OSError:
            pass
        pending.extend(children - discovered.keys())
    return discovered


def _fd_class(link: str) -> str:
    if link.startswith("pipe:"):
        return "pipe"
    if link.startswith("socket:"):
        return "socket"
    if link.startswith("memfd:"):
        return "memfd"
    if link == "anon_inode:[eventfd]":
        return "eventfd"
    if link == "anon_inode:[eventpoll]":
        return "eventpoll"
    if link.startswith("/dev/nvidia") or link.startswith("/dev/nvidia-cuda"):
        return "nvidia"
    if link.startswith("/dev/shm/") or link.startswith("/run/shm/"):
        return "shared_memory"
    return "other"


def _process_snapshot(pid: int, start_ticks: set[int]) -> dict[str, Any] | None:
    proc = _proc_dir(pid)
    try:
        if not proc.exists() or _process_start_ticks(pid) != start_ticks:
            return None
    except OSError:
        return None

    rss_kib = 0
    try:
        status = (proc / "status").read_text(encoding="utf-8")
    except OSError:
        status = ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    rss_kib = int(fields[1])
                except ValueError:
                    rss_kib = 0
            break

    fd_classes: dict[str, int] = {}
    fd_count = 0
    try:
        fd_entries = list((proc / "fd").iterdir())
    except OSError:
        fd_entries = []
    for fd in fd_entries:
        try:
            link = os.readlink(fd)
        except OSError:
            continue
        fd_count += 1
        label = _fd_class(link)
        fd_classes[label] = fd_classes.get(label, 0) + 1

    try:
        cmdline = (
            (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        )
    except OSError:
        cmdline = ""
    return {
        "pid": pid,
        "start_ticks": sorted(start_ticks),
        "rss_kib": rss_kib,
        "fd_count": fd_count,
        "fd_classes": dict(sorted(fd_classes.items())),
        "command": cmdline[:512],
    }


def _tree_snapshot(pid: int, known: dict[int, set[int]] | None = None) -> dict[str, Any]:
    discovered = known or _discover_descendants(pid)
    processes: list[dict[str, Any]] = []
    for child_pid, starts in sorted(discovered.items()):
        snapshot = _process_snapshot(child_pid, starts)
        if snapshot is not None:
            processes.append(snapshot)
    return {
        "process_count": len(processes),
        "rss_kib_total": sum(int(process["rss_kib"]) for process in processes),
        "fd_count_total": sum(int(process["fd_count"]) for process in processes),
        "fd_classes_total": _sum_dicts(
            process["fd_classes"]
            for process in processes
            if isinstance(process["fd_classes"], dict)
        ),
        "processes": processes,
    }


def _sum_dicts(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            result[key] = result.get(key, 0) + int(value)
    return dict(sorted(result.items()))


def _nvidia_smi_compute_apps() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps",
                "pid,process_name,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    names = ("pid", "process_name", "used_gpu_memory", "gpu_uuid")
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 4 and values != ["No running processes found"]:
            rows.append(dict(zip(names, values, strict=True)))
    return rows


def _gpu_memory_mib(value: str) -> int | None:
    fields = value.split()
    if len(fields) != 2 or fields[1] not in {"MiB", "MB"}:
        return None
    try:
        return int(fields[0])
    except ValueError:
        return None


def _gpu_snapshot(processes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tree_pids = {int(process["pid"]) for process in processes}
    rows = [row for row in _nvidia_smi_compute_apps() if int(row["pid"]) in tree_pids]
    memories = [_gpu_memory_mib(row["used_gpu_memory"]) for row in rows]
    return {
        "process_count": len(rows),
        "used_memory_mib_total": sum(value for value in memories if value is not None),
        "processes": rows,
    }


def _progress_snapshot(progress_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    console_log_path = (progress_dir / "soak-console.log").resolve()
    for path in progress_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == console_log_path:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {
        "file_count": len(files),
        "bytes_total": sum(int(item["bytes"]) for item in files),
        "max_mtime": max((float(item["mtime"]) for item in files), default=None),
    }


def _terminate_process_group(pid: int, process: subprocess.Popen[bytes] | None = None) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if process is not None:
            process.poll()
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process is not None:
        process.poll()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_snapshot(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=path,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_files": len(git("status", "--porcelain").splitlines()),
    }


def _gpu_device_snapshot() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    names = ("index", "uuid", "name", "driver_version")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(names):
            rows.append(dict(zip(names, values, strict=True)))
    return rows


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _software_snapshot() -> dict[str, str | None]:
    """Record monitor-process versions without importing accelerator packages."""
    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
    }


def workspace_snapshot(root: Path) -> dict[str, Any]:
    """Record source, pinned-sibling, GPU, and environment provenance."""
    manifest_path = root / "tensor_runtime_workspace.json"
    manifest = _load_json(manifest_path) or {}
    siblings: dict[str, Any] = {}
    for item in manifest.get("siblings", []):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        path_value = item.get("path")
        if isinstance(path_value, str):
            siblings[item["name"]] = _git_snapshot((root / path_value).resolve())
    return {
        "source": _git_snapshot(root),
        "workspace_manifest": manifest,
        "siblings": siblings,
        "gpu_devices": _gpu_device_snapshot(),
        "software": _software_snapshot(),
        "environment": {
            key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "UNILAB_LOCAL_UNISIM")
        },
    }


def _resource_summary(
    samples: Sequence[Mapping[str, Any]], residual_count: int, run_summary: Mapping[str, Any] | None
) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for sample in samples:
            value: Any = sample
            for key in path:
                if not isinstance(value, Mapping):
                    value = None
                    break
                value = value.get(key)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    process_counts = values(("process", "process_count"))
    rss = values(("process", "rss_kib_total"))
    gpu_memory = values(("gpu", "used_memory_mib_total"))
    eventfds = values(("process", "fd_classes_total", "eventfd"))
    shared_memory = values(("process", "fd_classes_total", "shared_memory"))
    progress_bytes = values(("progress", "bytes_total"))
    manifest = run_summary.get("runtime_manifest") if run_summary is not None else None
    flight = manifest.get("inference_flight") if isinstance(manifest, Mapping) else None
    budget = manifest.get("inference_memory_budget") if isinstance(manifest, Mapping) else None
    return {
        "sample_count": len(samples),
        "max_process_count": max(process_counts, default=None),
        "max_process_rss_kib": max(rss, default=None),
        "max_gpu_used_memory_mib": max(gpu_memory, default=None),
        "max_eventfd_count": max(eventfds, default=None),
        "max_shared_memory_fd_count": max(shared_memory, default=None),
        "max_progress_bytes": max(progress_bytes, default=None),
        "residual_process_count": residual_count,
        "ipc_event_count": budget.get("ipc_event_count") if isinstance(budget, Mapping) else None,
        "max_in_flight": flight.get("max_in_flight") if isinstance(flight, Mapping) else None,
        "max_publication_lag": (
            flight.get("max_publication_lag") if isinstance(flight, Mapping) else None
        ),
    }


def run_soak(
    *,
    command: Sequence[str],
    progress_dir: Path,
    output_path: Path,
    sample_interval_seconds: float = 5.0,
    startup_timeout_seconds: float = 600.0,
    stale_progress_seconds: float = 120.0,
    post_shutdown_grace_seconds: float = 5.0,
    min_duration_seconds: float = 0.0,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a command in its own process group and produce a soak artifact.

    ``progress_dir`` is an owner-controlled training log directory.  File
    growth is used as a backend-independent liveness signal; once progress is
    first observed, no file change for ``stale_progress_seconds`` fails the
    soak rather than allowing a deadlocked worker to run forever.
    """
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if startup_timeout_seconds <= 0:
        raise ValueError("startup_timeout_seconds must be positive")
    if stale_progress_seconds <= 0:
        raise ValueError("stale_progress_seconds must be positive")
    if post_shutdown_grace_seconds < 0:
        raise ValueError("post_shutdown_grace_seconds must be nonnegative")
    if min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be nonnegative")

    progress_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console_log_path = progress_dir / "soak-console.log"
    started_wall = time.time()
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    failure_reason: str | None = None
    process: subprocess.Popen[bytes] | None = None
    last_progress_change: float | None = None
    last_progress_bytes: int | None = None
    last_known_processes: list[dict[str, Any]] = []
    returncode: int | None = None

    with console_log_path.open("wb") as console_log:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(os.environ) if env is None else dict(env),
                stdout=console_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                now = time.monotonic()
                tree = _tree_snapshot(process.pid)
                progress = _progress_snapshot(progress_dir)
                last_known_processes = tree["processes"]
                if progress["bytes_total"] > 0:
                    if (
                        last_progress_bytes is None
                        or progress["bytes_total"] != last_progress_bytes
                    ):
                        last_progress_change = now
                        last_progress_bytes = int(progress["bytes_total"])
                    if (
                        last_progress_change is not None
                        and now - last_progress_change > stale_progress_seconds
                    ):
                        failure_reason = (
                            "training progress did not change for "
                            f"{now - last_progress_change:.1f}s"
                        )
                        break
                elif now - started > startup_timeout_seconds:
                    failure_reason = (
                        "training produced no progress artifact within "
                        f"{startup_timeout_seconds:.1f}s"
                    )
                    break
                samples.append(
                    {
                        "elapsed_s": round(now - started, 6),
                        "process": tree,
                        "gpu": _gpu_snapshot(tree["processes"]),
                        "progress": progress,
                    }
                )
                time.sleep(sample_interval_seconds)
        except KeyboardInterrupt:
            failure_reason = "soak was interrupted before the trainer completed"
        except Exception as exc:
            failure_reason = f"soak monitor failed: {type(exc).__name__}: {exc}"
        finally:
            if process is not None and process.poll() is None:
                if failure_reason is None:
                    failure_reason = "soak monitor exited before the trainer"
                _terminate_process_group(process.pid, process)
            if process is not None:
                returncode = process.wait()

    if failure_reason is None and returncode != 0:
        failure_reason = f"trainer exited with returncode {returncode}"
    if failure_reason is None and time.monotonic() - started < min_duration_seconds:
        failure_reason = f"soak completed before min_duration_seconds={min_duration_seconds:.1f}"

    if post_shutdown_grace_seconds:
        time.sleep(post_shutdown_grace_seconds)
    residual_processes = _residual_processes(last_known_processes)
    if failure_reason is None and residual_processes:
        pids = ", ".join(str(item["pid"]) for item in residual_processes)
        failure_reason = f"residual child processes after shutdown: {pids}"

    run_summary = _load_json(progress_dir / "run_summary.json")
    if failure_reason is None:
        if run_summary is None:
            failure_reason = "trainer did not write run_summary.json"
        elif run_summary.get("status") != "completed":
            failure_reason = (
                f"run_summary status is {run_summary.get('status')!r}, expected 'completed'"
            )
        else:
            manifest = run_summary.get("runtime_manifest")
            flight = manifest.get("inference_flight") if isinstance(manifest, dict) else None
            if isinstance(flight, dict):
                queue_depth = flight.get("queue_depth")
                publication_lag = flight.get("publication_lag")
                if queue_depth not in (None, 0) or publication_lag not in (None, 0):
                    failure_reason = (
                        "final inference flight is not drained: "
                        f"queue_depth={queue_depth!r}, publication_lag={publication_lag!r}"
                    )

    completed = failure_reason is None
    artifact: dict[str, Any] = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "status": "passed" if completed else "failed",
        "failure_reason": failure_reason,
        "command": list(command),
        "context": dict(metadata or {}),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
        "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.monotonic() - started, 6),
        "returncode": returncode,
        "monitor": {
            "sample_interval_s": sample_interval_seconds,
            "startup_timeout_s": startup_timeout_seconds,
            "stale_progress_s": stale_progress_seconds,
            "post_shutdown_grace_s": post_shutdown_grace_seconds,
            "min_duration_s": min_duration_seconds,
            "sample_count": len(samples),
        },
        "samples": samples,
        "resource_summary": _resource_summary(samples, len(residual_processes), run_summary),
        "post_shutdown": {
            "residual_processes": residual_processes,
        },
        "run_summary": run_summary,
        "console_log": str(console_log_path),
    }
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if not completed:
        raise SoakFailureError(f"{failure_reason}; soak artifact: {output_path}")
    return artifact


def _residual_processes(last_known: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    residual: list[dict[str, Any]] = []
    for item in last_known:
        pid = int(item["pid"])
        expected = {int(value) for value in item.get("start_ticks", [])}
        snapshot = _process_snapshot(pid, expected)
        if snapshot is not None:
            residual.append(snapshot)
    return residual


def build_g1_flashsac_mjwarp_command(
    *,
    num_envs: int,
    iterations: int,
    log_dir: Path,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build the owner-selected, single-process FlashSAC/MJWarp soak command."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    reserved = {
        "algo",
        "task",
        "training.sim_backend",
        "training.play_only",
        "training.no_play",
        "training.log_dir",
        "algo.num_envs",
        "algo.max_iterations",
    }
    for override in extra_overrides:
        key = override.split("=", 1)[0].strip().lstrip("+~")
        if key in reserved:
            raise ValueError(f"soak-owned Hydra override cannot be overridden: {key}")
    return build_command(
        mode="train",
        algo="flashsac",
        task="g1_motion_tracking",
        sim="mjwarp",
        overrides=(
            f"algo.num_envs={num_envs}",
            f"algo.max_iterations={iterations}",
            "training.no_play=true",
            f"training.log_dir={log_dir}",
            *extra_overrides,
        ),
    )
