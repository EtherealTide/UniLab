#!/usr/bin/env python3
"""Extract end-to-end off-policy benchmark metrics from TensorBoard event files.

unilab-rl 1.4.1 replaced the ``timing/*`` and lowercase ``perf/*`` tags with the
canonical schema in ``uni_rl.logging.metric_schema`` (see ``docs/metrics.md`` in
the unilab_rl repo). A current run records ``metric_schema_version = 1`` in
``run_summary.json`` and contains canonical ``Perf/`` / ``Train/`` scalar tags.
Historical event files are immutable and can be read only with the explicit
``--legacy-unversioned`` compatibility mode. That mode accepts recognized
pre-schema lowercase ``perf/*``, ``timing/*``, ``train/*``, and ``reward/*``
tags, never canonical tags, and never a newer schema version. Two canonical
fields are seconds where the retired tags were milliseconds:
``Perf/iteration_time`` (was ``perf/iter_ms``) and ``Perf/learning_time`` (was
``timing/learner_train_ms``). Rows whose source chart was retired without a
replacement report NaN on schema-v1 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

TAGS: dict[str, tuple[tuple[str, float], ...]] = {
    "iter_ms": (("Perf/iteration_time", 1000.0), ("perf/iter_ms", 1.0)),
    "steps_per_sec": (("Perf/total_fps", 1.0), ("perf/steps_per_sec", 1.0)),
    "collector_active_steps_per_sec": (("perf/collector_active_steps_per_sec", 1.0),),
    "collector_cycle_ms": (("perf/collector_cycle_ms", 1.0),),
    "collector_learner_action_wait_ms": (
        ("Perf/collector_learner_action_wait_ms", 1.0),
        ("timing/collector_learner_action_wait_ms", 1.0),
    ),
    "collector_replay_write_ms": (
        ("Perf/collector_replay_write_ms", 1.0),
        ("timing/collector_replay_write_ms", 1.0),
    ),
    "learner_collector_wait_ms": (
        ("Perf/learner_collector_wait_ms", 1.0),
        ("timing/learner_collector_wait_ms", 1.0),
    ),
    "learner_inference_ms": (
        ("Perf/learner_inference_ms", 1.0),
        ("timing/learner_inference_ms", 1.0),
    ),
    "learner_replay_batch_wait_ms": (
        ("Perf/learner_replay_batch_wait_ms", 1.0),
        ("timing/learner_replay_batch_wait_ms", 1.0),
    ),
    "learner_replay_sample_ms": (
        ("Perf/learner_replay_sample_ms", 1.0),
        ("timing/learner_replay_sample_ms", 1.0),
    ),
    "replay_ingress_h2d_submit_ms": (
        ("Perf/replay_ingress_h2d_submit_ms", 1.0),
        ("timing/replay_ingress_h2d_submit_ms", 1.0),
    ),
    "learner_collector_release_ms": (
        ("Perf/learner_collector_release_ms", 1.0),
        ("timing/learner_collector_release_ms", 1.0),
    ),
    "learner_train_ms": (("Perf/learning_time", 1000.0), ("timing/learner_train_ms", 1.0)),
}

_SUPPORTED_METRIC_SCHEMA_VERSION = 1
_ALL_TAG_CANDIDATES = tuple(tag for candidates in TAGS.values() for tag, _ in candidates)
_CANONICAL_TAGS = frozenset(
    tag for tag in _ALL_TAG_CANDIDATES if tag.startswith(("Perf/", "Train/"))
)
_LEGACY_TAGS = frozenset(_ALL_TAG_CANDIDATES) - _CANONICAL_TAGS


class MetricSchemaError(RuntimeError):
    """The TensorBoard/run-summary schema is missing, unsupported, or mixed."""


def _read_run_summary(event_file: Path) -> dict[str, Any] | None:
    """Read only the summary co-located with the selected event file."""
    summary_path = event_file.parent / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload: Any = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricSchemaError(f"cannot read run summary {summary_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MetricSchemaError(f"run summary must be a JSON object: {summary_path}")
    return payload


def _validate_metric_schema(
    summary: dict[str, Any] | None,
    scalar_tags: set[str],
    *,
    allow_legacy_unversioned: bool,
) -> bool:
    """Return whether extraction is restricted to the immutable legacy tags."""
    summary_has_schema_key = summary is not None and "metric_schema_version" in summary
    if allow_legacy_unversioned and summary_has_schema_key:
        schema_value = summary["metric_schema_version"]
        raise MetricSchemaError(
            "--legacy-unversioned cannot read a schema-stamped run summary; "
            f"metric_schema_version={schema_value!r}"
        )
    version = None if summary is None else summary.get("metric_schema_version")
    if version is not None or not allow_legacy_unversioned:
        if version is None:
            subject = "run summary is missing" if summary is not None else "no run summary has"
            raise MetricSchemaError(f"{subject} metric_schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise MetricSchemaError(f"metric_schema_version must be an integer, got {version!r}")
        if version != _SUPPORTED_METRIC_SCHEMA_VERSION:
            raise MetricSchemaError(
                f"unsupported metric_schema_version {version!r}; expected "
                f"{_SUPPORTED_METRIC_SCHEMA_VERSION}"
            )

        present_legacy = sorted(_LEGACY_TAGS & scalar_tags)
        if present_legacy:
            raise MetricSchemaError(
                "schema-v1 event files must not contain legacy tags: " + ", ".join(present_legacy)
            )
        if not _CANONICAL_TAGS & scalar_tags:
            raise MetricSchemaError("schema-v1 event file has no canonical Perf/ or Train/ tags")
        return False

    # The explicit historical reader is deliberately narrow. It cannot become a
    # fallback for unversioned future logs: a recognized legacy series is
    # mandatory and any canonical tag requires a schema-stamped summary.
    present_canonical = sorted(_CANONICAL_TAGS & scalar_tags)
    if present_canonical:
        raise MetricSchemaError(
            "canonical tags require metric_schema_version, found: " + ", ".join(present_canonical)
        )
    if not _LEGACY_TAGS & scalar_tags:
        raise MetricSchemaError(
            "--legacy-unversioned requires recognized pre-schema tags and no canonical tags"
        )
    return True


def _candidates_for_schema(
    candidates: tuple[tuple[str, float], ...], *, legacy_unversioned: bool
) -> tuple[tuple[str, float], ...]:
    return tuple(
        candidate
        for candidate in candidates
        if (candidate[0] in _LEGACY_TAGS) == legacy_unversioned
    )


def _find_event_file(log_dir: Path) -> Path | None:
    candidates = list(log_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise MetricSchemaError(
            f"expected one event file under {log_dir}, found {len(candidates)}; "
            "pass one run directory instead of a parent directory"
        )
    return candidates[0]


def _average_last(scalars: list[event_accumulator.ScalarEvent], n: int) -> float:
    values = [s.value for s in scalars]
    if not values:
        return float("nan")
    return sum(values[-n:]) / len(values[-n:])


def _extract_row(
    ea: event_accumulator.EventAccumulator,
    candidates: tuple[tuple[str, float], ...],
    last: int,
) -> float:
    available = set(ea.Tags()["scalars"])
    for tag, scale in candidates:
        if tag in available:
            return _average_last(ea.Scalars(tag), last) * scale
    return float("nan")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--last", type=int, default=20, help="Average over the last N iterations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--legacy-unversioned",
        action="store_true",
        help=(
            "read only an immutable pre-schema event file with recognized legacy tags; "
            "canonical or schema-stamped runs are rejected"
        ),
    )
    args = parser.parse_args(argv)

    try:
        event_file = _find_event_file(args.log_dir)
    except MetricSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if event_file is None:
        print(f"No event file found under {args.log_dir}", file=sys.stderr)
        return 1

    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    try:
        summary = _read_run_summary(event_file)
        legacy_unversioned = _validate_metric_schema(
            summary,
            set(ea.Tags()["scalars"]),
            allow_legacy_unversioned=args.legacy_unversioned,
        )
    except MetricSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows = [
        (
            label,
            _extract_row(
                ea,
                _candidates_for_schema(candidates, legacy_unversioned=legacy_unversioned),
                args.last,
            ),
        )
        for label, candidates in TAGS.items()
    ]

    if args.json:
        print(json.dumps({label: value for label, value in rows}, indent=2))
    else:
        for label, value in rows:
            print(f"{label:40} {value:10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
