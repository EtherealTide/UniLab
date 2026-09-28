from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

from unilab.training.soak import (
    SoakFailureError,
    _tree_snapshot,
    build_g1_flashsac_mjwarp_command,
    run_soak,
    workspace_snapshot,
)


def _valid_replay_ingress() -> dict[str, object]:
    return {
        "ingress_depth": 2,
        "ingress_slot_rows": 1024,
        "published_sequence": 100,
        "release_sequence": 100,
        "occupancy": 0,
        "high_water_occupancy": 2,
        "backpressure_waits": 3,
        "backpressure_wait_s": 0.25,
        "early_returns": 0,
        "dropped_batches": 0,
        "closed_returns": 0,
        "stop_returns": 0,
    }


def _completed_summary() -> dict[str, Any]:
    return {
        "status": "completed",
        "total_env_steps": 102_400,
        "metric_schema_version": 1,
        "runtime_manifest": {
            "schema_version": 1,
            "inference_ring_capacity": 1,
            "collector_metrics_interval": 100,
            "collector_backend_device": "cuda:0",
            "inference_flight": {
                "queue_depth": 0,
                "publication_lag": 0,
                "max_in_flight": 1,
                "max_publication_lag": 1,
            },
            "replay_ingress": _valid_replay_ingress(),
        },
    }


def _run_with_summary(tmp_path: Path, summary: object) -> dict[str, object]:
    progress_dir = tmp_path / "run"
    command = [
        sys.executable,
        "-c",
        (
            "import json, pathlib, time; "
            f"d=pathlib.Path({str(progress_dir)!r}); d.mkdir(parents=True, exist_ok=True); "
            "(d / 'events').write_text('progress'); time.sleep(0.01); "
            f"(d / 'run_summary.json').write_text(json.dumps({summary!r}))"
        ),
    ]
    return run_soak(
        command=command,
        progress_dir=progress_dir,
        output_path=tmp_path / "artifact.json",
        sample_interval_seconds=0.02,
        startup_timeout_seconds=2.0,
        stale_progress_seconds=1.0,
        post_shutdown_grace_seconds=0.0,
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
    summary = _completed_summary()

    artifact = _run_with_summary(tmp_path, summary)

    assert artifact["status"] == "passed"
    assert artifact["returncode"] == 0
    assert artifact["run_summary"]["status"] == "completed"
    assert artifact["post_shutdown"]["residual_processes"] == []
    assert artifact["resource_summary"]["sample_count"] >= 1
    assert artifact["resource_summary"]["max_process_count"] >= 1
    assert artifact["schema_version"] == "0.2.0"
    assert artifact["resource_summary"]["replay_ingress_depth"] == 2
    assert artifact["resource_summary"]["replay_ingress_high_water_occupancy"] == 2
    assert artifact["resource_summary"]["replay_ingress_final_occupancy"] == 0
    assert artifact["resource_summary"]["replay_ingress_dropped_batches"] == 0
    assert artifact["resource_summary"]["replay_ingress_early_returns"] == 0
    saved = json.loads((tmp_path / "artifact.json").read_text())
    assert saved["status"] == "passed"
    assert saved["monitor"]["sample_count"] >= 1


def test_workspace_snapshot_records_software_provenance(tmp_path: Path) -> None:
    context = workspace_snapshot(tmp_path)

    assert context["software"]["python"] == platform.python_version()
    assert isinstance(context["software"]["torch"], (str, type(None)))


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
    summary = _completed_summary()
    summary["runtime_manifest"]["inference_flight"]["queue_depth"] = 1

    with pytest.raises(SoakFailureError, match="final inference flight is not drained"):
        _run_with_summary(tmp_path, summary)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (lambda ingress: ingress.pop("occupancy"), "occupancy must be an integer"),
        (lambda ingress: ingress.update(ingress_depth="2"), "ingress_depth must be an integer"),
        (lambda ingress: ingress.update(ingress_depth=True), "ingress_depth must be an integer"),
        (
            lambda ingress: ingress.update(published_sequence=100.5),
            "published_sequence must be an integer",
        ),
        (
            lambda ingress: ingress.update(backpressure_wait_s="nan"),
            "backpressure_wait_s must be finite",
        ),
        (
            lambda ingress: ingress.update(backpressure_wait_s=-0.1),
            "backpressure_wait_s must be nonnegative",
        ),
        (lambda ingress: ingress.update(ingress_depth=0), "ingress_depth must be positive"),
        (
            lambda ingress: ingress.update(ingress_slot_rows=0),
            "ingress_slot_rows must be positive",
        ),
        (
            lambda ingress: ingress.update(high_water_occupancy=3),
            "high_water_occupancy exceeds ingress_depth",
        ),
        (
            lambda ingress: ingress.update(
                occupancy=1, release_sequence=99, high_water_occupancy=1
            ),
            "final replay ingress occupancy is nonzero",
        ),
        (
            lambda ingress: ingress.update(dropped_batches=1, early_returns=1, closed_returns=1),
            "replay ingress dropped batches during normal run",
        ),
        (
            lambda ingress: ingress.update(release_sequence=101),
            "release_sequence exceeds published_sequence",
        ),
        (
            lambda ingress: ingress.update(occupancy=1),
            "occupancy does not match publication sequences",
        ),
        (
            lambda ingress: ingress.update(published_sequence=99, release_sequence=99),
            "published_sequence does not match total_env_steps",
        ),
        (
            lambda ingress: ingress.update(closed_returns=1),
            "shutdown return counters disagree",
        ),
    ],
)
def test_soak_monitor_fails_closed_on_invalid_replay_ingress(
    tmp_path: Path,
    mutation,
    expected_failure: str,
) -> None:
    replay_ingress = _valid_replay_ingress()
    mutation(replay_ingress)
    summary = _completed_summary()
    summary["runtime_manifest"]["replay_ingress"] = replay_ingress

    with pytest.raises(SoakFailureError, match=expected_failure):
        _run_with_summary(tmp_path, summary)

    artifact = json.loads((tmp_path / "artifact.json").read_text())
    assert artifact["status"] == "failed"
    assert expected_failure in artifact["failure_reason"]


@pytest.mark.parametrize("missing_field", ["runtime_manifest", "replay_ingress"])
def test_soak_monitor_requires_final_replay_ingress(tmp_path: Path, missing_field: str) -> None:
    summary = _completed_summary()
    runtime_manifest = summary["runtime_manifest"]
    if missing_field == "replay_ingress":
        runtime_manifest.pop("replay_ingress")
    if missing_field == "runtime_manifest":
        summary.pop("runtime_manifest")

    with pytest.raises(SoakFailureError, match=f"missing {missing_field}"):
        _run_with_summary(tmp_path, summary)


@pytest.mark.parametrize("replay_ingress", [None, [], "invalid", {}])
def test_soak_monitor_rejects_malformed_replay_ingress(
    tmp_path: Path, replay_ingress: object
) -> None:
    summary = _completed_summary()
    summary["runtime_manifest"]["replay_ingress"] = replay_ingress

    with pytest.raises(SoakFailureError, match="missing replay_ingress|must be an integer"):
        _run_with_summary(tmp_path, summary)


def test_soak_monitor_requires_total_env_steps_for_replay_publication_count(tmp_path: Path) -> None:
    summary = _completed_summary()
    summary.pop("total_env_steps")

    with pytest.raises(SoakFailureError, match="total_env_steps must be an integer"):
        _run_with_summary(tmp_path, summary)


@pytest.mark.parametrize("metric_schema_version", [None, 0, 2, "1", 1.0, True])
def test_soak_monitor_fails_closed_on_invalid_metric_schema(
    tmp_path: Path, metric_schema_version: object
) -> None:
    summary = _completed_summary()
    if metric_schema_version is None:
        summary.pop("metric_schema_version")
    else:
        summary["metric_schema_version"] = metric_schema_version

    with pytest.raises(SoakFailureError, match="metric_schema_version"):
        _run_with_summary(tmp_path, summary)


@pytest.mark.parametrize("runtime_schema_version", [None, 0, 2, "1", 1.0, True])
def test_soak_monitor_fails_closed_on_invalid_runtime_manifest_schema(
    tmp_path: Path, runtime_schema_version: object
) -> None:
    summary = _completed_summary()
    manifest = summary["runtime_manifest"]
    if runtime_schema_version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = runtime_schema_version

    with pytest.raises(SoakFailureError, match="runtime_manifest.schema_version"):
        _run_with_summary(tmp_path, summary)


@pytest.mark.parametrize(
    "missing_field",
    ["inference_ring_capacity", "collector_metrics_interval", "collector_backend_device"],
)
def test_soak_monitor_requires_completed_runtime_manifest_stable_fields(
    tmp_path: Path, missing_field: str
) -> None:
    summary = _completed_summary()
    summary["runtime_manifest"].pop(missing_field)

    with pytest.raises(SoakFailureError, match=missing_field):
        _run_with_summary(tmp_path, summary)


@pytest.mark.skipif(not Path("/proc/self").exists(), reason="requires Linux procfs")
def test_soak_tree_snapshot_counts_current_process() -> None:
    snapshot = _tree_snapshot(os.getpid())
    assert snapshot["process_count"] >= 1
    assert any(process["pid"] == os.getpid() for process in snapshot["processes"])
