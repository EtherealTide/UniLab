"""Guard the #2441 promotion of cross-module private names to public ones."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = (ROOT / "src", ROOT / "tests", ROOT / "scripts")

_MANAGERS_PACKAGE = ROOT / "src" / "unilab" / "managers"
_LOCOMOTION_COMMON_MANAGER_TERMS = (
    ROOT / "src" / "unilab" / "tasks" / "locomotion" / "common" / "manager_terms.py"
)

# Names retired by the #2441 public promotion. ``_command``/``_real`` stay
# private inside the motion_tracking/g1 owner modules, and ``_root_body_name``
# remains private on ``EntityData`` and task term classes, so they are not
# listed here.
RETIRED_TOKENS = (
    "_tensor_read_plan",
    "_tensor_reset_pose_bounds",
    "_tensor_reset_velocity_bounds",
    "_tensor_reset_bounds_key",
    "__call_reset__",
    "_update_command_reset",
    "_build_fixed_variant_plan",
    "_require_fixed_variant_support",
    "_build_play_args",
    "_compose_interactive_config",
    "_parse_interactive_cli",
    "_TEST_LOG_ROOT_ENV",
    "_normalize_load_run",
    "_validated_positive_int",
    "_resolve_render_play_model_files",
    "_quat_apply_inverse",
    "_read_plan",
    "_root_body_view",
    "_world_to_body_vector",
    "managers._types",
)

_RETIRED_PATTERNS = tuple(
    re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"\b") for token in RETIRED_TOKENS
)


def _iter_python_files() -> list[Path]:
    return sorted(
        path
        for root in SCAN_ROOTS
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path != Path(__file__).resolve()
    )


def test_managers_types_module_is_public() -> None:
    assert (_MANAGERS_PACKAGE / "types.py").exists()
    assert not (_MANAGERS_PACKAGE / "_types.py").exists()


def test_retired_private_names_do_not_reappear() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        text = path.read_text(encoding="utf-8")
        for pattern, token in zip(_RETIRED_PATTERNS, RETIRED_TOKENS, strict=True):
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{token}")
    assert offenders == [], f"retired private names reappeared: {offenders}"


def test_observations_and_locomotion_common_imports_stay_public() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            is_observations = node.module == "unilab.envs.mdp.observations"
            is_locomotion_common_terms = node.module in (
                "unilab.tasks.locomotion.common.manager_terms",
                "manager_terms",
            ) and (
                node.module.startswith("unilab.")
                or (node.level == 1 and path.parent == _LOCOMOTION_COMMON_MANAGER_TERMS.parent)
            )
            if not (is_observations or is_locomotion_common_terms):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno}:{alias.name}"
                    )
    assert violations == [], f"cross-module private imports remain: {violations}"
