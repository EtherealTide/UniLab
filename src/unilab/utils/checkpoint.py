"""Pure-filesystem log root and checkpoint resolution helpers."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

TEST_LOG_ROOT_ENV = "UNILAB_TEST_LOG_ROOT"


def get_latest_run(log_dir: str | Path) -> Path | None:
    """Return the lexicographically latest run directory under a task log root."""
    base_dir = Path(log_dir)
    if not base_dir.exists():
        return None
    runs = sorted(path for path in base_dir.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def get_latest_checkpoint(run_dir: str | Path, *, suffix: str = ".pt") -> Path | None:
    """Return the latest model checkpoint inside a run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    def _iteration(path: Path) -> int:
        stem_parts = path.stem.split("_", 1)
        if len(stem_parts) != 2:
            return -1
        try:
            return int(stem_parts[1])
        except ValueError:
            return -1

    model_files = [
        path
        for path in run_path.iterdir()
        if path.is_file() and path.name.startswith("model_") and path.suffix == suffix
    ]
    if not model_files:
        return None
    return max(model_files, key=_iteration)


def normalize_load_run(load_run: str | int | PathLike[str]) -> str:
    return str(load_run)


def normalize_checkpoint_value(value: object) -> str | None:
    """Normalize a raw checkpoint selector value; sentinel values map to ``None``."""
    if value is None:
        return None
    text = str(value)
    return None if text in {"", "-1", "None", "null"} else text


def resolve_checkpoint_path(
    task_log_root: str | Path,
    load_run: str | int | PathLike[str],
    *,
    checkpoint: str | int | None = None,
    suffix: str = ".pt",
) -> tuple[Path | None, Path | None]:
    """Resolve a checkpoint from a task log root.

    ``load_run`` selects the run: ``"-1"`` picks the latest run under
    ``task_log_root``; any other value is first tried as a direct path and
    then relative to ``task_log_root``, and may name either a run directory
    or a checkpoint file. ``checkpoint`` optionally selects one checkpoint
    inside the resolved run (an iteration number or a filename).

    Returns ``(checkpoint_path, run_dir)``. A resolved run that lacks the
    requested (or any) checkpoint yields ``(None, run_dir)``; an
    unresolvable run yields ``(None, None)``.
    """
    base_dir = Path(task_log_root)
    selected_run = normalize_load_run(load_run)

    run_dir: Path | None
    if selected_run == "-1":
        run_dir = get_latest_run(base_dir)
    else:
        candidate = Path(selected_run)
        if not candidate.exists():
            candidate = base_dir / selected_run
        if candidate.is_file():
            return candidate, candidate.parent
        run_dir = candidate if candidate.is_dir() else None

    if run_dir is None:
        return None, None

    checkpoint_path: Path | None
    if checkpoint is not None:
        checkpoint_name = (
            f"model_{checkpoint}{suffix}" if str(checkpoint).isdigit() else str(checkpoint)
        )
        checkpoint_path = run_dir / checkpoint_name
        return (checkpoint_path, run_dir) if checkpoint_path.exists() else (None, run_dir)

    checkpoint_path = get_latest_checkpoint(run_dir, suffix=suffix)
    return (checkpoint_path, run_dir) if checkpoint_path is not None else (None, run_dir)
