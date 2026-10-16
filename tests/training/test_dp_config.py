"""Hydra-side data-parallel configuration tests (owner: unilab.training.dp_config).

Merged from UniLab's former ``tests/ipc/test_dp_launcher.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.rl.runtime.distributed.affinity import resolve_collector_cpu_ids
from unilab.training.dp_config import apply_dp_rank_config, reject_removed_device_config

_ROOT = Path(__file__).parent.parent.parent
_CONF_DIR = _ROOT / "src" / "unilab" / "conf"


def _offpolicy_cfg(overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    normalized = list(overrides or [])
    if not any(override.startswith("task=") for override in normalized):
        normalized.append("task=g1_walk_flat/mujoco")
    with initialize_config_dir(config_dir=str(_CONF_DIR / "sac"), version_base="1.3"):
        return compose("config", overrides=normalized)


# ---------------------------------------------------------------------------
# removed training.devices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("devices_cfg", [None, []])
def test_removed_device_config_accepts_empty_defaults(devices_cfg):
    reject_removed_device_config(devices_cfg)


def test_removed_device_config_fails_closed():
    with pytest.raises(ValueError, match="training.devices was removed"):
        reject_removed_device_config((0, 1))


# ---------------------------------------------------------------------------
# Hydra compose surface
# ---------------------------------------------------------------------------


def test_offpolicy_config_no_longer_exposes_devices():
    cfg = _offpolicy_cfg()

    assert "devices" not in cfg.training


# ---------------------------------------------------------------------------
# apply_dp_rank_config / N=1 equivalence
# ---------------------------------------------------------------------------


def test_apply_dp_rank_config_maps_rank_to_device_and_seed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    cfg = _offpolicy_cfg(["algo.seed=42"])
    base_seed = int(cfg.algo.seed)
    device = apply_dp_rank_config(cfg, rank=1)
    assert device == "cuda:0"
    assert "device" not in cfg.training
    assert int(cfg.algo.seed) == base_seed + 1


def test_apply_dp_rank_config_rank_zero_keeps_seed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    cfg = _offpolicy_cfg(["algo.seed=42"])
    assert apply_dp_rank_config(cfg, rank=0) == "cuda:0"
    assert int(cfg.algo.seed) == 42


def test_offpolicy_config_dp_collector_cpu_ids_defaults_to_null():
    cfg = _offpolicy_cfg()
    assert cfg.training.dp_collector_cpu_ids is None


def test_offpolicy_config_dp_collector_cpu_ids_compose():
    cfg = _offpolicy_cfg(["training.dp_collector_cpu_ids=[[0,1],[2,3]]"])
    explicit = OmegaConf.to_container(cfg.training.dp_collector_cpu_ids, resolve=True)
    assert explicit == [[0, 1], [2, 3]]
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=explicit) == [2, 3]
