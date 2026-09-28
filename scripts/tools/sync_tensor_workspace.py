#!/usr/bin/env python3
"""Materialize the pinned local tensor-runtime sibling workspace.

The integration profile intentionally uses relative editable dependencies.  This
script makes those dependencies reproducible without publishing packages: it can
clone missing siblings, fetch and checkout the commits recorded in the manifest,
and verify that an existing workspace is clean and pinned to those commits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Sibling:
    name: str
    repository: str
    ref: str
    commit: str
    path: Path
    sentinel_env: str | None


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"`{' '.join(command)}` failed ({result.returncode}): {detail}")
    return result


def _require_str(mapping: dict[str, Any], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must define non-empty string field '{key}'")
    return value


def load_manifest(path: Path, workspace_root: Path) -> list[Sibling]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read tensor workspace manifest {path}: {error}") from error

    if payload.get("schema_version") != 1:
        raise ValueError("tensor workspace manifest schema_version must be 1")
    if payload.get("profile") != "local-editable":
        raise ValueError("tensor workspace manifest must declare the local-editable profile")

    records = payload.get("siblings")
    if not isinstance(records, list) or not records:
        raise ValueError("tensor workspace manifest must contain at least one sibling")

    siblings: list[Sibling] = []
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each tensor workspace sibling must be an object")
        name = _require_str(record, "name", "sibling")
        if name in names:
            raise ValueError(f"duplicate tensor workspace sibling: {name}")
        names.add(name)
        relative_path = _require_str(record, "path", f"sibling {name}")
        path = Path(relative_path)
        if not path.is_absolute():
            path = workspace_root / path
        sentinel_env = record.get("sentinel_env")
        if sentinel_env is not None and (
            not isinstance(sentinel_env, str) or not sentinel_env.isidentifier()
        ):
            raise ValueError(f"sibling {name} has invalid sentinel_env")
        siblings.append(
            Sibling(
                name=name,
                repository=_require_str(record, "repository", f"sibling {name}"),
                ref=_require_str(record, "ref", f"sibling {name}"),
                commit=_require_str(record, "commit", f"sibling {name}"),
                path=path.resolve(),
                sentinel_env=sentinel_env,
            )
        )
    return siblings


def _git_state(path: Path) -> tuple[str | None, bool]:
    if not (path / ".git").exists():
        return None, False
    head = _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()
    status = _run(["git", "status", "--porcelain"], cwd=path).stdout.splitlines()
    return head, bool(status)


def _checkout_sibling(sibling: Sibling, *, sync: bool) -> None:
    if not (sibling.path / ".git").exists():
        if not sync:
            raise RuntimeError(
                f"{sibling.name} checkout is missing at {sibling.path}; "
                "run `make sync-workspace` or pass --sync"
            )
        sibling.path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", sibling.repository, str(sibling.path)])

    head, dirty = _git_state(sibling.path)
    if dirty:
        raise RuntimeError(
            f"{sibling.name} checkout at {sibling.path} is dirty; "
            "commit or stash sibling work before changing the pinned workspace"
        )
    if head == sibling.commit:
        print(f"{sibling.name}: {head} (pinned)")
        return
    if not sync:
        raise RuntimeError(
            f"{sibling.name} is at {head}, expected {sibling.commit}; "
            "run `make sync-workspace` or pass --sync"
        )
    _run(["git", "fetch", "origin", sibling.commit], cwd=sibling.path)
    _run(["git", "checkout", "--detach", sibling.commit], cwd=sibling.path)
    print(f"{sibling.name}: {head} -> {sibling.commit}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "tensor_runtime_workspace.json",
        help="workspace manifest (default: repository root manifest)",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="root used to resolve relative sibling paths (default: UniLab root)",
    )
    parser.add_argument(
        "--sync", action="store_true", help="clone/fetch and checkout pinned commits"
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print shell export lines for declared sentinels after validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    workspace_root = args.workspace_root.resolve()
    try:
        siblings = load_manifest(manifest, workspace_root)
        for sibling in siblings:
            _checkout_sibling(sibling, sync=args.sync)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.print_env:
        for sibling in siblings:
            if sibling.sentinel_env is not None:
                print(f"export {sibling.sentinel_env}={sibling.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
