"""CPU affinity wiring tests for the Motrix backend (#962).

Covers the UniLab side of the contract: ``EnvCfg.cpu_ids`` flows through
``env_backend_kwargs``/``create_backend`` into the unisim Motrix adapter,
which validates the block on the cold path and pins MotrixSim's shared
worker pool through ``motrixsim.init_thread_pool``. The pool is process-wide
and can only be initialized once, so passthrough is asserted with a mocked
initializer; the behavioral per-thread pinning check lives in unisim
(``tests/adapters/motrix/test_cpu_ids.py``).
"""

import os
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("motrixsim", reason="motrixsim not installed")

import motrixsim
from unisim.backend.motrix.backend import MotrixBackend

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg

_MODEL_FILE = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "mjlab_cartpole" / "cartpole.xml"
)
_NUM_ENVS = 4
_BASE_NAME = "cart"


def _cpu_block() -> list[int]:
    """CPU ids usable by this process; slim CI runners may expose only one."""
    get = getattr(os, "sched_getaffinity", None)
    return sorted(get(0))[:2] if get is not None else [0, 1]


def test_env_backend_kwargs_maps_cpu_ids():
    cfg = EnvCfg(cpu_ids=[2, 3])
    assert env_backend_kwargs(cfg)["cpu_ids"] == [2, 3]
    assert env_backend_kwargs(EnvCfg())["cpu_ids"] is None


def test_create_backend_routes_cpu_ids_to_motrix():
    ids = _cpu_block()
    with mock.patch.object(motrixsim, "init_thread_pool") as init:
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=_MODEL_FILE),
            _NUM_ENVS,
            0.01,
            base_name=_BASE_NAME,
            cpu_ids=ids,
        )
    try:
        assert isinstance(backend, MotrixBackend)
        assert backend.cpu_ids == tuple(ids)
        init.assert_called_once_with(core_ids=ids)
    finally:
        backend.close()


def test_create_backend_routes_cpu_ids_from_env_cfg():
    ids = _cpu_block()
    cfg = EnvCfg(cpu_ids=ids)
    with mock.patch.object(motrixsim, "init_thread_pool") as init:
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=_MODEL_FILE),
            _NUM_ENVS,
            0.01,
            base_name=_BASE_NAME,
            **env_backend_kwargs(cfg),
        )
    try:
        assert backend.cpu_ids == tuple(ids)
        init.assert_called_once_with(core_ids=ids)
    finally:
        backend.close()


@pytest.mark.parametrize("cpu_ids", ([0, 0], [-1], []))
def test_backend_rejects_invalid_cpu_ids_on_cold_path(cpu_ids):
    with pytest.raises(ValueError):
        MotrixBackend(
            SceneCfg(model_file=_MODEL_FILE),
            num_envs=_NUM_ENVS,
            sim_dt=0.01,
            base_name=_BASE_NAME,
            cpu_ids=cpu_ids,
        )


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux-only affinity check")
def test_unavailable_cpu_id_rejected_on_cold_path():
    unavailable = max(os.sched_getaffinity(0)) + 4096
    with pytest.raises(ValueError, match="not available"):
        MotrixBackend(
            SceneCfg(model_file=_MODEL_FILE),
            num_envs=_NUM_ENVS,
            sim_dt=0.01,
            base_name=_BASE_NAME,
            cpu_ids=[unavailable],
        )


def test_default_path_leaves_worker_pool_untouched():
    with mock.patch.object(motrixsim, "init_thread_pool") as init:
        backend = create_backend(
            "motrix",
            SceneCfg(model_file=_MODEL_FILE),
            _NUM_ENVS,
            0.01,
            base_name=_BASE_NAME,
        )
    try:
        assert backend.cpu_ids is None
        init.assert_not_called()
    finally:
        backend.close()
