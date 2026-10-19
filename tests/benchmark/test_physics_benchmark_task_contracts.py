from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Callable

import pytest


def _load_optional_benchmark_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"{name} requires an unavailable benchmark dependency: {exc}")


def _isaacgym_module() -> ModuleType:
    return _load_optional_benchmark_module(
        "scripts.benchmark.physics.benchmark_physics_step_isaacgym"
    )


def _normalize(module: ModuleType) -> Callable[[str], str]:
    return module._normalize_task_name


def _isaacsim_module() -> ModuleType:
    return _load_optional_benchmark_module(
        "scripts.benchmark.physics.benchmark_physics_step_isaacsim"
    )


@pytest.mark.parametrize("task_id", ["go2_joystick_flat", "g1_walk_flat"])
def test_mujoco_benchmarks_accept_canonical_task_ids(task_id: str) -> None:
    for module_name in (
        "scripts.benchmark.physics.benchmark_physics_step_mujoco_warp",
        "scripts.benchmark.physics.benchmark_physics_step_mj_step",
        "scripts.benchmark.physics.benchmark_physics_step_genesis",
    ):
        module = _load_optional_benchmark_module(module_name)
        assert module.normalize_locomotion_task_id(task_id) == task_id


@pytest.mark.parametrize("task_id", ["go2_joystick_flat", "g1_walk_flat"])
def test_isaac_benchmarks_accept_canonical_task_ids(task_id: str) -> None:
    assert _normalize(_isaacgym_module())(task_id) == task_id
    assert _normalize(_isaacsim_module())(task_id) == task_id


@pytest.mark.parametrize(
    "removed_input",
    [
        "Go2JoystickFlat",
        "G1WalkFlat",
        "task=go2_joystick_flat/mujoco",
        "task=g1_walk_flat/isaacgym",
        "go2_joystick_flat/isaacsim",
        "go2",
        "g1",
    ],
)
def test_isaac_benchmarks_reject_removed_task_forms(removed_input: str) -> None:
    with pytest.raises(ValueError, match="Pass one of those owner task ids verbatim"):
        _normalize(_isaacgym_module())(removed_input)

    with pytest.raises(ValueError, match="Pass one of those owner task ids verbatim"):
        _normalize(_isaacsim_module())(removed_input)


def test_isaac_benchmark_help_only_exposes_tasks() -> None:
    for module in (
        _isaacgym_module(),
        _isaacsim_module(),
    ):
        source = inspect.getsource(module)

        assert "--tasks" in source
        assert "--task " not in source


def test_isaacgym_runtime_paths_must_be_configured_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isaacgym_module()
    legacy_deps_name = "UNILAB_ISAACGYM_" + "DEPS_ROOT"
    legacy_models_name = "UNILAB_ISAACGYM_" + "MODELS_ROOT"

    monkeypatch.delenv("UNILAB_BENCHMARK_HOLOSOMA_DEPS", raising=False)
    monkeypatch.delenv("UNILAB_BENCHMARK_MODELS_ROOT", raising=False)
    monkeypatch.setenv(legacy_deps_name, "/legacy/runtime")
    monkeypatch.setenv(legacy_models_name, "/legacy/models")

    with pytest.raises(RuntimeError, match="UNILAB_BENCHMARK_HOLOSOMA_DEPS"):
        module._required_env_path("UNILAB_BENCHMARK_HOLOSOMA_DEPS")
    with pytest.raises(RuntimeError, match="UNILAB_BENCHMARK_MODELS_ROOT"):
        module._required_env_path("UNILAB_BENCHMARK_MODELS_ROOT")


def test_isaac_runtime_import_does_not_require_machine_paths() -> None:
    assert _isaacgym_module().gymapi is None
    assert _isaacsim_module()._isaac_modules == {}


def test_isaacsim_runtime_paths_must_be_configured_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _isaacsim_module()

    monkeypatch.delenv("ISAACSIM_ROOT", raising=False)
    monkeypatch.delenv("ISAACSIM_USD_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="ISAACSIM_ROOT"):
        module._required_env_path("ISAACSIM_ROOT")
    with pytest.raises(RuntimeError, match="ISAACSIM_USD_ROOT"):
        module._required_env_path("ISAACSIM_USD_ROOT")
