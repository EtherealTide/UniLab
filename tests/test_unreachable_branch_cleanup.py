"""Guards for the unreachable platform-branch cleanup in issue #2381."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.rl.runtime.ipc.replay_pipelines.device import require_offpolicy_replay_device

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "unilab"


def test_offpolicy_replay_device_gate_keeps_xpu_unreachable() -> None:
    with pytest.raises(ValueError, match="replay device to equal the learner device"):
        require_offpolicy_replay_device("xpu")


def test_offpolicy_learners_have_no_xpu_amp_whitelist() -> None:
    for relative_path in ("rl/algos/sac/learner.py", "rl/algos/flash_sac/learner.py"):
        source = (SRC_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'in ("cuda", "xpu")' not in source


def test_isaac_backend_symbols_stay_behind_owner_boundaries() -> None:
    owner_boundaries = {
        "base/backend_factory.py",
        "base/base.py",
        "base/config_adapter.py",
        "base/process_device.py",
        "base/registry.py",
        "cli.py",
    }
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"isaacgym|isaacsim", source) is None:
            continue
        relative = path.relative_to(SRC_ROOT).as_posix()
        assert relative in owner_boundaries or relative.startswith("tasks/")

    registry.ensure_registries()
    assert registry._envs["G1WalkFlat"].supports_sim_backend("isaacgym")
    assert registry._envs["G1WalkFlat"].supports_sim_backend("isaacsim")


def test_manager_env_has_no_cuda_ipc_warm_branch_for_shelved_backends() -> None:
    source = (SRC_ROOT / "envs" / "manager_based_rl_env.py").read_text(encoding="utf-8")
    assert "_warm_external_cuda_ipc_views" not in source
    assert "isaacgym" not in source


def test_env_cfg_owns_no_unsupported_isaac_candidate_fields() -> None:
    field_names = {field.name for field in fields(EnvCfg)}
    unsupported_candidates = {
        "isaacsim_solver_position_iteration_count",
        "isaacsim_solver_velocity_iteration_count",
        "isaacsim_bounce_threshold_velocity",
        "isaacsim_contact_offset",
        "isaacsim_rest_offset",
        "isaacsim_max_depenetration_velocity",
        "isaacsim_gpu_max_rigid_contact_count",
        "isaacsim_gpu_max_rigid_patch_count",
    }
    assert not [name for name in field_names if name in unsupported_candidates]


def test_registry_reward_resolver_has_no_legacy_target() -> None:
    source = (SRC_ROOT / "base" / "registry.py").read_text(encoding="utf-8")
    assert "reward_config" not in source


def test_sac_actor_explore_has_no_bool_positional_shim() -> None:
    for relative_path in ("rl/algos/sac/learner.py", "rl/algos/flash_sac/network.py"):
        source = (SRC_ROOT / relative_path).read_text(encoding="utf-8")
        assert "isinstance(dones, bool)" not in source
