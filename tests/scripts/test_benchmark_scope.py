from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
BENCHMARK_ROOT = ROOT / "scripts" / "benchmark"
MARKER = "HISTORICAL SHELVED-BACKEND BENCHMARK (#1811)"
HISTORICAL_BENCHMARKS = (
    "benchmark_drake_performance.py",
    "physics/benchmark_isaacgym_fixed_variants.py",
    "physics/benchmark_motrix_body_state_ab.py",
    "physics/benchmark_motrix_set_state_ab.py",
    "physics/benchmark_mujoco_vs_motrix.py",
    "physics/benchmark_physics_step_isaacgym.py",
    "physics/benchmark_physics_step_isaacsim.py",
    "physics/benchmark_physics_step_motrixsim.py",
    "physics/benchmark_sim.py",
    "superdex_go2_collector.py",
    "torch_env/g1_flashsac_backend.py",
)


def test_historical_adapter_benchmarks_declare_issue_1811_scope() -> None:
    offenders: list[str] = []
    for relative in HISTORICAL_BENCHMARKS:
        path = BENCHMARK_ROOT / relative
        if MARKER not in path.read_text(encoding="utf-8"):
            offenders.append(relative)
    assert offenders == [], f"missing historical benchmark marker: {offenders}"
