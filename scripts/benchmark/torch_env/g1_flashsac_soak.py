#!/usr/bin/env python3
"""Run and monitor a single-GPU G1 FlashSAC/MJWarp training soak."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unilab.training.soak import (
    build_g1_flashsac_mjwarp_command,
    run_soak,
    runtime_snapshot,
)

ROOT_DIR = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=0.0,
        help="fail a successful trainer that finishes before this duration",
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--stale-progress-seconds", type=float, default=300.0)
    parser.add_argument("--post-shutdown-grace-seconds", type=float, default=10.0)
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="additional Hydra override; soak-owned keys are rejected",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    command = build_g1_flashsac_mjwarp_command(
        num_envs=args.num_envs,
        iterations=args.iterations,
        log_dir=args.log_dir,
        extra_overrides=args.extra_override,
    )
    artifact = run_soak(
        command=command,
        progress_dir=args.log_dir,
        output_path=args.output,
        sample_interval_seconds=args.sample_interval_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        stale_progress_seconds=args.stale_progress_seconds,
        post_shutdown_grace_seconds=args.post_shutdown_grace_seconds,
        min_duration_seconds=args.min_duration_seconds,
        cwd=ROOT_DIR,
        metadata=runtime_snapshot(ROOT_DIR),
    )
    print(json.dumps(artifact["monitor"], indent=2))


if __name__ == "__main__":
    main()
