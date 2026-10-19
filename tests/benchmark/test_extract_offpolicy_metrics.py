from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scripts.benchmark.rl import extract_offpolicy_metrics as extract

torch = pytest.importorskip("torch")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


def _write_tfevents(log_dir: Path, series: dict[str, list[float]]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for tag, values in series.items():
            for step, value in enumerate(values):
                writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def _write_run_summary(log_dir: Path, **overrides: object) -> None:
    payload = {"status": "completed", "metric_schema_version": 1, **overrides}
    (log_dir / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_json(log_dir: Path, capsys: pytest.CaptureFixture) -> dict[str, float]:
    assert extract.main([str(log_dir), "--json", "--last", "20"]) == 0
    return json.loads(capsys.readouterr().out)


def test_extract_new_schema_tags_with_unit_conversion(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(
        tmp_path,
        {
            "Perf/iteration_time": [1.0, 1.5],  # seconds
            "Perf/learning_time": [0.4, 0.6],  # seconds
            "Perf/total_fps": [100.0, 300.0],
            "Perf/learner_collector_wait_ms": [5.0, 7.0],
        },
    )
    rows = _run_json(tmp_path, capsys)
    assert rows["iter_ms"] == pytest.approx(1_250.0)
    assert rows["learner_train_ms"] == pytest.approx(500.0)
    assert rows["steps_per_sec"] == pytest.approx(200.0)
    assert rows["learner_collector_wait_ms"] == pytest.approx(6.0)
    # Retired charts without a canonical replacement stay NaN on new runs.
    assert math.isnan(rows["collector_active_steps_per_sec"])
    assert math.isnan(rows["collector_cycle_ms"])


def test_extract_legacy_tags_fall_back_unscaled(tmp_path: Path, capsys) -> None:
    # Unversioned summaries are historical only and require the explicit mode.
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    _write_tfevents(
        tmp_path,
        {
            "perf/iter_ms": [1_000.0, 1_500.0],
            "timing/learner_train_ms": [400.0, 600.0],
            "perf/steps_per_sec": [100.0, 300.0],
            "perf/collector_active_steps_per_sec": [50.0, 70.0],
        },
    )
    assert extract.main([str(tmp_path), "--json", "--last", "20", "--legacy-unversioned"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows["iter_ms"] == pytest.approx(1_250.0)
    assert rows["learner_train_ms"] == pytest.approx(500.0)
    assert rows["steps_per_sec"] == pytest.approx(200.0)
    assert rows["collector_active_steps_per_sec"] == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("tag", "row", "value"),
    [
        ("perf/iter_ms", "iter_ms", 1_250.0),
        ("timing/learner_train_ms", "learner_train_ms", 500.0),
    ],
)
def test_extract_legacy_mode_accepts_each_recognized_family(
    tmp_path: Path, capsys, tag: str, row: str, value: float
) -> None:
    _write_tfevents(tmp_path, {tag: [value]})

    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 0

    parsed = json.loads(capsys.readouterr().out)
    assert parsed[row] == pytest.approx(value)


@pytest.mark.parametrize(
    "tag",
    [
        "reward/mean",
        "reward/mean_ep100",
        "train/dp_sync_time",
    ],
)
def test_extract_legacy_mode_rejects_undocumented_reader_families(
    tmp_path: Path, capsys, tag: str
) -> None:
    # dcbd09e3 evidence: despite their global pre-schema migration history,
    # reward/train return-summary tags are not recognized extraction inputs in
    # this reader. #2416 may narrow or extend that scope explicitly; this test
    # must move with any such behavior change.
    _write_tfevents(tmp_path, {tag: [1.0]})

    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 1
    assert "--legacy-unversioned requires recognized pre-schema tags" in capsys.readouterr().err


def test_extract_legacy_mode_rejects_newer_schema_stamp(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path, metric_schema_version=2)
    _write_tfevents(tmp_path, {"perf/steps_per_sec": [100.0]})

    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 1
    assert "cannot read a schema-stamped" in capsys.readouterr().err


@pytest.mark.parametrize("legacy_tag", ["perf/iter_ms", "timing/learner_train_ms"])
def test_extract_current_mode_rejects_mixed_recognized_families(
    tmp_path: Path, capsys, legacy_tag: str
) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(tmp_path, {"Perf/total_fps": [100.0], legacy_tag: [1.0]})

    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "must not contain legacy tags" in capsys.readouterr().err


def test_extract_current_mode_ignores_tags_outside_reader_scope(tmp_path: Path, capsys) -> None:
    # This pins the current narrow reader scope: reward/train tags are outside
    # its recognized-tag set and therefore do not participate in mixed rejection.
    _write_run_summary(tmp_path)
    _write_tfevents(tmp_path, {"Perf/total_fps": [100.0], "reward/mean": [1.0]})

    assert extract.main([str(tmp_path), "--json"]) == 0


def test_extract_does_not_fall_back_from_canonical_to_legacy(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(
        tmp_path,
        {
            "Perf/total_fps": [999.0],
            "perf/steps_per_sec": [111.0],
        },
    )
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "must not contain legacy tags" in capsys.readouterr().err


def test_extract_missing_event_file_is_an_error(tmp_path: Path, capsys) -> None:
    assert extract.main([str(tmp_path)]) == 1
    assert "No event file found" in capsys.readouterr().err


def test_extract_rejects_multiple_event_files(tmp_path: Path, capsys) -> None:
    _write_tfevents(tmp_path / "run1", {"Perf/total_fps": [1.0]})
    _write_tfevents(tmp_path / "run2", {"Perf/total_fps": [2.0]})
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "expected one event file" in capsys.readouterr().err


def test_extract_uses_only_summary_co_located_with_event(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(tmp_path / "rank0", {"Perf/total_fps": [1.0]})
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "metric_schema_version" in capsys.readouterr().err


def test_extract_requires_schema_stamp_for_canonical_tags(tmp_path: Path, capsys) -> None:
    _write_tfevents(tmp_path, {"Perf/total_fps": [100.0]})
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "metric_schema_version" in capsys.readouterr().err


def test_extract_rejects_missing_summary_by_default(tmp_path: Path, capsys) -> None:
    _write_tfevents(tmp_path, {"perf/steps_per_sec": [100.0]})
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "metric_schema_version" in capsys.readouterr().err


def test_extract_rejects_unsupported_schema_version(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path, metric_schema_version=2)
    _write_tfevents(tmp_path, {"Perf/total_fps": [100.0]})
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "unsupported metric_schema_version 2" in capsys.readouterr().err


def test_extract_rejects_mixed_canonical_and_legacy_tags(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(
        tmp_path,
        {
            "Perf/total_fps": [100.0],
            "perf/steps_per_sec": [200.0],
        },
    )
    assert extract.main([str(tmp_path), "--json"]) == 1
    assert "must not contain legacy tags" in capsys.readouterr().err


def test_extract_legacy_mode_rejects_schema_stamped_summary(tmp_path: Path, capsys) -> None:
    _write_run_summary(tmp_path)
    _write_tfevents(tmp_path, {"perf/steps_per_sec": [100.0]})
    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 1
    assert "cannot read a schema-stamped" in capsys.readouterr().err


def test_extract_legacy_mode_rejects_null_schema_stamp(tmp_path: Path, capsys) -> None:
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"status": "completed", "metric_schema_version": None}),
        encoding="utf-8",
    )
    _write_tfevents(tmp_path, {"perf/steps_per_sec": [100.0]})
    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 1
    assert "cannot read a schema-stamped" in capsys.readouterr().err


def test_extract_legacy_mode_rejects_canonical_tags_without_schema(tmp_path: Path, capsys) -> None:
    _write_tfevents(
        tmp_path,
        {
            "Perf/total_fps": [100.0],
            "perf/steps_per_sec": [200.0],
        },
    )
    assert extract.main([str(tmp_path), "--json", "--legacy-unversioned"]) == 1
    assert "canonical tags require metric_schema_version" in capsys.readouterr().err


def test_extract_legacy_mode_accepts_missing_summary(tmp_path: Path, capsys) -> None:
    _write_tfevents(tmp_path, {"perf/steps_per_sec": [100.0, 300.0]})
    assert extract.main([str(tmp_path), "--json", "--last", "20", "--legacy-unversioned"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows["steps_per_sec"] == pytest.approx(200.0)
