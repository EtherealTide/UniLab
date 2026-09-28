from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from unilab.training.soak import (
    SoakFailureError,
    _tree_snapshot,
    build_g1_flashsac_mjwarp_command,
    run_soak,
)


def test_g1_flashsac_soak_command_uses_owner_route(tmp_path: Path) -> None:
    command = build_g1_flashsac_mjwarp_command(
        num_envs=16,
        iterations=3,
        log_dir=tmp_path / "run",
        extra_overrides=("algo.seed=7",),
    )

    assert command[1].endswith("train_flashsac.py")
    assert "task=g1_motion_tracking/mjwarp" in command
    assert "algo.num_envs=16" in command
    assert "algo.max_iterations=3" in command
    assert "training.no_play=true" in command
    assert "algo.seed=7" in command


def test_g1_flashsac_soak_command_rejects_owned_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="soak-owned Hydra override"):
        build_g1_flashsac_mjwarp_command(
            num_envs=16,
            iterations=3,
            log_dir=tmp_path / "run",
            extra_overrides=("algo.max_iterations=100",),
        )


def test_soak_monitor_accepts_completed_run(tmp_path: Path) -> None:
    progress_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-c",
        (
            "import json, pathlib, time; "
            f"d=pathlib.Path({str(progress_dir)!r}); d.mkdir(parents=True, exist_ok=True); "
            "(d / 'events').write_text('progress'); time.sleep(0.05); "
            "(d / 'run_summary.json').write_text(json.dumps({"
            "'status': 'completed', "
            "'runtime_manifest': {'inference_flight': {'queue_depth': 0, "
            "'publication_lag': 0}}}"
            "))"
        ),
    ]

    artifact = run_soak(
        command=command,
        progress_dir=progress_dir,
        output_path=tmp_path / "artifact.json",
        sample_interval_seconds=0.02,
        startup_timeout_seconds=2.0,
        stale_progress_seconds=1.0,
        post_shutdown_grace_seconds=0.0,
    )

    assert artifact["status"] == "passed"
    assert artifact["returncode"] == 0
    assert artifact["run_summary"]["status"] == "completed"
    assert artifact["post_shutdown"]["residual_processes"] == []
    assert artifact["resource_summary"]["sample_count"] >= 1
    assert artifact["resource_summary"]["max_process_count"] >= 1
    saved = json.loads((tmp_path / "artifact.json").read_text())
    assert saved["status"] == "passed"
    assert saved["monitor"]["sample_count"] >= 1


def test_soak_monitor_fails_closed_on_stale_progress(tmp_path: Path) -> None:
    progress_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib, time; "
            f"d=pathlib.Path({str(progress_dir)!r}); d.mkdir(parents=True, exist_ok=True); "
            "(d / 'events').write_text('stalled'); time.sleep(5)"
        ),
    ]

    with pytest.raises(SoakFailureError, match="did not change"):
        run_soak(
            command=command,
            progress_dir=progress_dir,
            output_path=tmp_path / "artifact.json",
            sample_interval_seconds=0.05,
            startup_timeout_seconds=2.0,
            stale_progress_seconds=0.01,
            post_shutdown_grace_seconds=0.0,
        )

    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert artifact["status"] == "failed"
    assert "did not change" in artifact["failure_reason"]
    assert artifact["post_shutdown"]["residual_processes"] == []


def test_soak_monitor_fails_when_summary_is_missing(tmp_path: Path) -> None:
    progress_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib; "
            f"d=pathlib.Path({str(progress_dir)!r}); d.mkdir(parents=True, exist_ok=True); "
            "(d / 'events').write_text('progress')"
        ),
    ]

    with pytest.raises(SoakFailureError, match="run_summary"):
        run_soak(
            command=command,
            progress_dir=progress_dir,
            output_path=tmp_path / "artifact.json",
            sample_interval_seconds=0.02,
            startup_timeout_seconds=2.0,
            stale_progress_seconds=1.0,
            post_shutdown_grace_seconds=0.0,
        )


def test_soak_monitor_fails_when_final_flight_is_not_drained(tmp_path: Path) -> None:
    progress_dir = tmp_path / "run"
    summary = {
        "status": "completed",
        "runtime_manifest": {"inference_flight": {"queue_depth": 1, "publication_lag": 0}},
    }
    command = [
        sys.executable,
        "-c",
        (
            "import json, pathlib; "
            f"d=pathlib.Path({str(progress_dir)!r}); d.mkdir(parents=True, exist_ok=True); "
            f"(d / 'events').write_text('progress'); "
            f"(d / 'run_summary.json').write_text(json.dumps({summary!r}))"
        ),
    ]

    with pytest.raises(SoakFailureError, match="final inference flight is not drained"):
        run_soak(
            command=command,
            progress_dir=progress_dir,
            output_path=tmp_path / "artifact.json",
            sample_interval_seconds=0.02,
            startup_timeout_seconds=2.0,
            stale_progress_seconds=1.0,
            post_shutdown_grace_seconds=0.0,
        )


@pytest.mark.skipif(not Path("/proc/self").exists(), reason="requires Linux procfs")
def test_soak_tree_snapshot_counts_current_process() -> None:
    snapshot = _tree_snapshot(os.getpid())
    assert snapshot["process_count"] >= 1
    assert any(process["pid"] == os.getpid() for process in snapshot["processes"])
