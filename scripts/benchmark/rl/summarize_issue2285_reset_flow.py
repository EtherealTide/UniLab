#!/usr/bin/env python3
"""Summarize issue #2285 reset-flow evidence artifacts.

The script validates authoritative E2E summaries, computes paired A/B/C
statistics, extracts collector reset-row histograms, and emits the decision
record. It intentionally does not import environment runtime code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VARIANTS = ("A", "B", "C")
T_CRITICAL_DF2 = 2.776
RESET_BINS = (
    (0, 0),
    (1, 16),
    (17, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
    (1025, 2047),
    (2048, 2048),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _paired_stats(baseline: list[float], experimental: list[float]) -> dict[str, Any]:
    if len(baseline) != len(experimental) or not baseline:
        raise ValueError("paired statistics require aligned non-empty samples")
    ratios = [value / base for value, base in zip(experimental, baseline)]
    mean = statistics.fmean(ratios)
    if len(ratios) > 1:
        stdev = statistics.stdev(ratios)
        half_ci = T_CRITICAL_DF2 * stdev / math.sqrt(len(ratios))
    else:
        stdev = 0.0
        half_ci = 0.0
    return {
        "paired_repeats": len(ratios),
        "baseline_mean": statistics.fmean(baseline),
        "experimental_mean": statistics.fmean(experimental),
        "paired_ratio_mean": mean,
        "paired_delta_percent_mean": (mean - 1.0) * 100.0,
        "paired_delta_percent_std": stdev * 100.0,
        "ci95_half_width_percent": half_ci * 100.0,
        "ci95_low_percent": (mean - 1.0) * 100.0 - half_ci * 100.0,
        "ci95_high_percent": (mean - 1.0) * 100.0 + half_ci * 100.0,
        "samples": [
            {
                "repeat": index + 1,
                "baseline": baseline[index],
                "experimental": experimental[index],
                "ratio": ratio,
                "delta_percent": (ratio - 1.0) * 100.0,
            }
            for index, ratio in enumerate(ratios)
        ],
    }


def summarize_e2e(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for summary_path in sorted(root.rglob("run_summary.json")):
        scenario_dir = summary_path.parent.parent
        run_dir = summary_path.parent
        if scenario_dir == root or "-" not in run_dir.name:
            continue
        scenario = scenario_dir.name
        variant, repeat_text = run_dir.name.split("-", 1)
        if (
            variant not in VARIANTS
            or not repeat_text.startswith("r")
            or not repeat_text[1:].isdigit()
        ):
            continue
        repeat = int(repeat_text[1:])
        data = _read_json(summary_path)
        required = {
            "status": "completed",
            "completed_iterations": 160,
            "metric_schema_version": 1,
            "tail_window_iterations": 100,
            "tail_iteration_count": 100,
        }
        errors = [
            f"{key}={data.get(key)!r}, expected {expected!r}"
            for key, expected in required.items()
            if data.get(key) != expected
        ]
        tail = data.get("tail_env_steps_per_sec")
        final = data.get("final_env_steps_per_sec")
        if not isinstance(tail, (int, float)) or tail <= 0:
            errors.append("tail_env_steps_per_sec must be positive")
        row = {
            "scenario": scenario,
            "variant": variant,
            "repeat": repeat,
            "valid": not errors,
            "errors": "; ".join(errors),
            "tail_env_steps_per_sec": (float(tail) if isinstance(tail, (int, float)) else None),
            "final_env_steps_per_sec": (float(final) if isinstance(final, (int, float)) else None),
            "total_env_steps": data.get("total_env_steps"),
            "path": str(summary_path),
        }
        rows.append(row)
        if not errors:
            grouped[scenario][variant].append(float(tail))

    paired: dict[str, Any] = {}
    for scenario in sorted(grouped):
        baseline = grouped[scenario].get("A", [])
        if not baseline:
            continue
        paired[scenario] = {}
        for variant in ("B", "C"):
            experimental = grouped[scenario].get(variant, [])
            if not experimental:
                continue
            try:
                paired[scenario][variant] = _paired_stats(baseline, experimental)
            except ValueError as exc:
                paired[scenario][variant] = {"error": str(exc)}
    return rows, paired


def _bin_name(lower: int, upper: int) -> str:
    return str(lower) if lower == upper else f"{lower}_{upper}"


def summarize_collector(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    for json_path in sorted(root.rglob("*.json")):
        payload = _read_json(json_path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            continue
        for index, result in enumerate(payload["results"]):
            case = result.get("case", {})
            variant = str(case.get("variant", "default"))
            if variant not in VARIANTS:
                continue
            timings = result.get("env_step_timing_ms_per_vector_step", {})
            stat = timings.get("reset_done_count")
            if not isinstance(stat, dict):
                continue
            samples = [float(value) for value in stat.get("samples_ms", []) if value >= 0]
            if not samples:
                continue
            counts = Counter()
            for value in samples:
                for lower, upper in RESET_BINS:
                    if lower <= int(value) <= upper:
                        counts[_bin_name(lower, upper)] += 1
                        break
            histogram = {
                "source": str(json_path),
                "result_index": index,
                "task": case.get("task"),
                "variant": variant,
                "reset_steps": len(samples),
                "reset_mean": statistics.fmean(samples),
                "reset_median": statistics.median(samples),
                "reset_p05": _percentile(samples, 0.05),
                "reset_p95": _percentile(samples, 0.95),
                "reset_min": min(samples),
                "reset_max": max(samples),
                "zero_step_fraction": samples.count(0.0) / len(samples),
                "full_reset_step_fraction": samples.count(2048.0) / len(samples),
                "bins": {name: counts[name] for name in (_bin_name(*pair) for pair in RESET_BINS)},
            }
            histograms.append(histogram)
            rows.append(
                {
                    "source": str(json_path),
                    "result_index": index,
                    "task": case.get("task"),
                    "variant": variant,
                    "collector_active_steps_per_sec": result.get("collector_active_steps_per_sec"),
                    "tail_active_steps_per_sec": result.get("tail_active_steps_per_sec"),
                    "env_step_ms": result.get("phase_ms_per_vector_step", {})
                    .get("env_step_ms", {})
                    .get("mean_ms"),
                    "update_state_ms": timings.get("update_state_ms", {}).get("mean_ms"),
                    "reset_done_ms": timings.get("reset_done_ms", {}).get("mean_ms"),
                    "reset_done_count": stat.get("mean_ms"),
                }
            )
    return rows, histograms


def build_decision(paired: dict[str, Any]) -> dict[str, Any]:
    canonical = {}
    heavy = {}
    for workload in ("sac", "flash"):
        scenario = workload
        canonical_b = paired.get(scenario, {}).get("B", {}).get("paired_delta_percent_mean")
        heavy_b = paired.get(f"{workload}-heavy", {}).get("B", {}).get("paired_delta_percent_mean")
        canonical[workload] = canonical_b
        heavy[workload] = heavy_b
    canonical_ok = all(
        isinstance(value, (int, float)) and value >= -2.0 for value in canonical.values()
    )
    heavy_ok = all(isinstance(value, (int, float)) and value >= -5.0 for value in heavy.values())
    conclusion = "remove_dynamic_rows" if canonical_ok and heavy_ok else "owner_decision_or_reject"
    return {
        "schema_version": "issue2285/decision/v1",
        "primary_metric": "run_summary.json:tail_env_steps_per_sec",
        "predeclared_thresholds": {
            "canonical_accept_percent": -2.0,
            "canonical_owner_decision_range_percent": [-5.0, -2.0],
            "canonical_reject_percent": -5.0,
            "reset_heavy_accept_percent": -5.0,
            "reset_heavy_owner_decision_range_percent": [-10.0, -5.0],
            "reset_heavy_reject_percent": -10.0,
        },
        "canonical_b_vs_a_delta_percent": canonical,
        "reset_heavy_b_vs_a_delta_percent": heavy,
        "conclusion": conclusion,
        "c_used_for": "upper-bound diagnostic only",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=Path, required=True)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    e2e_rows, paired = summarize_e2e(args.e2e_root)
    collector_rows, histograms = summarize_collector(args.collector_root)
    decision = build_decision(paired)
    manifest = {
        "schema_version": "issue2285/manifest/v1",
        "issue": 2285,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variants": {
            "A": "dynamic_rows",
            "B": "mask_only",
            "C": "mask_full_refresh",
        },
        "authoritative_metric": "tail_env_steps_per_sec",
        "tail_window_iterations": 100,
        "invalid_e2e_runs": [row for row in e2e_rows if not row["valid"]],
        "artifacts": {
            "e2e": str(args.e2e_root),
            "collector": str(args.collector_root),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "e2e.csv", e2e_rows)
    _write_json(args.out_dir / "paired_statistics.json", paired)
    _write_csv(args.out_dir / "collector.csv", collector_rows)
    _write_csv(args.out_dir / "reset_histogram.csv", histograms)
    _write_json(args.out_dir / "decision.json", decision)
    _write_json(args.out_dir / "manifest.json", manifest)
    print(f"Wrote summaries to {args.out_dir}")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
