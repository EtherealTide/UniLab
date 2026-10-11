from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.benchmark.rl.summarize_issue2285_reset_flow import (
    _bin_name,
    build_decision,
    summarize_collector,
    summarize_e2e,
)


def _write_summary(path: Path, tail: float, *, status: str = "completed") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "completed_iterations": 160,
                "total_env_steps": 348160,
                "metric_schema_version": 1,
                "tail_env_steps_per_sec": tail,
                "tail_window_iterations": 100,
                "tail_iteration_count": 100,
                "final_env_steps_per_sec": tail - 1000.0,
            }
        ),
        encoding="utf-8",
    )


def test_e2e_summary_validates_and_pairs_interleaved_repeats(tmp_path: Path) -> None:
    _write_summary(tmp_path / "sac/A-r1/run_summary.json", 100.0)
    _write_summary(tmp_path / "sac/B-r1/run_summary.json", 110.0)
    _write_summary(tmp_path / "sac/A-r2/run_summary.json", 120.0)
    _write_summary(tmp_path / "sac/B-r2/run_summary.json", 126.0)
    _write_summary(tmp_path / "sac/A-r3/run_summary.json", 90.0, status="failed")

    rows, paired = summarize_e2e(tmp_path)
    assert [row["valid"] for row in rows].count(True) == 4
    assert [row["valid"] for row in rows].count(False) == 1
    stats = paired["sac"]["B"]
    assert stats["paired_delta_percent_mean"] == pytest.approx(7.5)
    assert stats["ci95_low_percent"] < stats["paired_delta_percent_mean"]
    assert stats["ci95_high_percent"] > stats["paired_delta_percent_mean"]


def test_collector_histogram_uses_predeclared_bins(tmp_path: Path) -> None:
    payload = {
        "results": [
            {
                "case": {"task": "task", "variant": "A"},
                "collector_active_steps_per_sec": 100.0,
                "tail_active_steps_per_sec": 100.0,
                "phase_ms_per_vector_step": {
                    "env_step_ms": {"mean_ms": 1.0},
                },
                "env_step_timing_ms_per_vector_step": {
                    "update_state_ms": {"mean_ms": 0.4},
                    "reset_done_ms": {"mean_ms": 0.2},
                    "reset_done_count": {
                        "mean_ms": 3.0,
                        "samples_ms": [0.0, 1.0, 64.0, 2048.0],
                    },
                },
            }
        ]
    }
    (tmp_path / "collector.json").write_text(json.dumps(payload), encoding="utf-8")
    rows, histograms = summarize_collector(tmp_path)
    histogram = histograms[0]
    assert histogram["reset_steps"] == 4
    assert histogram["bins"][_bin_name(0, 0)] == 1
    assert histogram["bins"][_bin_name(1, 16)] == 1
    assert histogram["bins"][_bin_name(17, 64)] == 1
    assert histogram["bins"][_bin_name(2048, 2048)] == 1
    assert rows[0]["reset_done_count"] == 3.0


def test_decision_accepts_only_both_canonical_and_heavy_workloads() -> None:
    paired = {
        "sac": {"B": {"paired_delta_percent_mean": 1.0}},
        "flash": {"B": {"paired_delta_percent_mean": -1.0}},
        "sac-heavy": {"B": {"paired_delta_percent_mean": -4.0}},
        "flash-heavy": {"B": {"paired_delta_percent_mean": -1.0}},
    }
    assert build_decision(paired)["conclusion"] == "remove_dynamic_rows"
    paired["flash-heavy"]["B"]["paired_delta_percent_mean"] = -6.0
    paired["flash"]["B"]["paired_delta_percent_mean"] = -6.0
    assert build_decision(paired)["conclusion"] == "owner_decision_or_reject"
