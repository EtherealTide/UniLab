"""Architecture boundary for backend-private and removed API usage."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "unilab"

# Factory/process owners may translate public owner configuration into backend
# construction inputs. Runtime/task capability dispatch is the #1811 target.
_NAME_DISPATCH_EXEMPT_FILES = {
    Path("base/backend_factory.py"),
    Path("base/env_factory.py"),
    Path("base/process_device.py"),
    Path("scripts/train_offpolicy.py"),
}

_MIGRATION_DEBT_NODES = frozenset()

REMOVED_BACKEND_METHODS = frozenset(
    {
        "step",
        "reset",
        "get_state",
        "set_state",
        "get_base_pos",
        "get_base_quat",
        "get_base_lin_vel",
        "get_base_ang_vel",
        "get_dof_pos",
        "get_dof_vel",
        "get_body_pos_w",
        "get_body_quat_w",
        "get_body_pose_w",
        "get_body_lin_vel_w",
        "get_body_vel_w",
        "get_body_ang_vel_w",
        "get_body_state_w",
        "copy_body_state_w",
        "get_body_pose_w_rows",
        "get_body_lin_vel_w_rows",
        "get_body_ang_vel_w_rows",
        "get_body_pos_b",
        "get_body_quat_b",
        "get_body_lin_vel_b",
        "get_body_ang_vel_b",
        "get_sensor_data",
        "get_sensor_data_rows",
        "get_sensor_data_batch",
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    kind: str
    detail: str


def _iter_source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def find_violations() -> list[Violation]:
    violations: list[Violation] = []
    backend_names = {"backend", "_backend"}
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                receiver = node.value
                if (
                    node.attr in REMOVED_BACKEND_METHODS
                    and isinstance(receiver, ast.Name)
                    and receiver.id in backend_names
                ):
                    violations.append(
                        Violation(
                            relative,
                            node.lineno,
                            "removed-backend-api",
                            f"{relative}:{node.attr}",
                        )
                    )
                if (
                    node.attr.startswith("_")
                    and isinstance(receiver, ast.Name)
                    and receiver.id in backend_names
                ):
                    detail = f"{relative}:{node.attr}"
                    if detail not in _MIGRATION_DEBT_NODES:
                        violations.append(
                            Violation(relative, node.lineno, "private-backend-attribute", detail)
                        )
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and len(node.test.ops) == 1
            ):
                left = node.test.left
                right = node.test.comparators[0]
                if (
                    isinstance(left, ast.Attribute)
                    and left.attr == "backend_type"
                    and isinstance(right, ast.Constant)
                    and isinstance(right.value, str)
                    and relative not in _NAME_DISPATCH_EXEMPT_FILES
                ):
                    violations.append(
                        Violation(
                            relative,
                            node.lineno,
                            "backend-name-dispatch",
                            f"{relative}:{right.value}",
                        )
                    )
    return violations


def test_runtime_and_tasks_do_not_probe_backend_private_state() -> None:
    assert find_violations() == []


def test_migration_exception_list_remains_explicit() -> None:
    assert not _MIGRATION_DEBT_NODES


def test_unisim_public_backend_api_has_no_legacy_numpy_lifecycle() -> None:
    from unisim.backend.base import SimBackend

    assert not REMOVED_BACKEND_METHODS.intersection(SimBackend.__dict__)
