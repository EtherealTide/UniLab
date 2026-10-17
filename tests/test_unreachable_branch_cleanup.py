"""Guards for the unreachable platform-branch cleanup in issue #2381."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.rl.runtime.ipc.replay_pipelines.gpu_resident import require_offpolicy_replay_device

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "unilab"


def test_offpolicy_replay_device_gate_keeps_xpu_unreachable() -> None:
    with pytest.raises(ValueError, match="CUDA or MPS learner device"):
        require_offpolicy_replay_device("xpu")


def test_offpolicy_learners_have_no_xpu_amp_whitelist() -> None:
    for relative_path in ("rl/algos/sac/learner.py", "rl/algos/flash_sac/learner.py"):
        source = (SRC_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'in ("cuda", "xpu")' not in source


def test_shelved_backend_symbols_are_confined_to_the_registry_list() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"isaacgym|isaacsim", source) is None:
            continue
        assert path.relative_to(SRC_ROOT).as_posix() == "base/registry.py"

    assert registry._SHELVED_SIM_BACKENDS == ("isaacgym", "isaacsim")


def test_manager_env_has_no_cuda_ipc_warm_branch_for_shelved_backends() -> None:
    source = (SRC_ROOT / "envs" / "manager_based_rl_env.py").read_text(encoding="utf-8")
    assert "_warm_external_cuda_ipc_views" not in source
    assert "isaacgym" not in source


def test_env_cfg_owns_no_shelved_backend_fields() -> None:
    field_names = {field.name for field in fields(EnvCfg)}
    assert not [name for name in field_names if name.startswith(("isaacgym_", "isaacsim_"))]


def test_registry_reward_resolver_has_no_legacy_target() -> None:
    source = (SRC_ROOT / "base" / "registry.py").read_text(encoding="utf-8")
    assert "reward_config" not in source


def test_sac_actor_explore_has_no_bool_positional_shim() -> None:
    source = (SRC_ROOT / "rl" / "algos" / "sac" / "learner.py").read_text(encoding="utf-8")
    assert "isinstance(dones, bool)" not in source
