from __future__ import annotations

import ast
from pathlib import Path

from scripts.benchmark.env import benchmark_env_step_phase_cpu as bench


def test_phase_benchmark_only_consumes_public_step_timing() -> None:
    source = Path(bench.__file__).read_text()
    tree = ast.parse(source)
    private_accesses: list[str] = []
    private_assignments: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            private_accesses.append(node.attr)
        if (
            isinstance(node, (ast.Assign, ast.AugAssign))
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "wrap"
        ):
            private_assignments.append(ast.dump(node))

    assert not private_accesses
    assert not private_assignments
    assert bench.PHASE_CPU_KEYS == (
        "apply_action_cpu_ms",
        "step_core_cpu_ms",
        "update_state_cpu_ms",
        "reset_done_cpu_ms",
        "env_step_other_cpu_ms",
    )
