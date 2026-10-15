"""Data-parallel launch and rank-supervision tests (owner: rl.runtime.distributed.launcher).

Merged from UniLab's former ``tests/ipc/test_dp_launcher.py`` and unilab_rl @
3654e4b (unilab-rl 1.4.15) ``tests/ipc/test_dp_launcher.py``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import unilab.rl.runtime.distributed.launcher as launcher
from unilab.rl.runtime.distributed.context import (
    UNILAB_DP_LOG_DIR,
    UNILAB_DP_RANK,
    UNILAB_DP_WORLD_SIZE,
)
from unilab.rl.runtime.distributed.launcher import DpRankSupervisor, launch_torchrun_workers


class _FakePopen:
    """Minimal subprocess.Popen stand-in with a scriptable exit code."""

    instances: list["_FakePopen"] = []
    # When True, wait() on a still-running child makes it exit cleanly
    # (simulates a rank that finishes during the normal-exit grace window).
    wait_completes: bool = False
    interrupt_exits: bool = True

    def __init__(self, argv, env=None, **kwargs):
        self.argv = list(argv)
        self.env = dict(env or {})
        self.start_new_session = bool(kwargs.pop("start_new_session", False))
        assert not kwargs
        self.pid = 10_000 + len(_FakePopen.instances)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.signals: list[int] = []
        _FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def send_signal(self, signum):
        self.signals.append(signum)
        if signum == signal.SIGINT and self.interrupt_exits:
            self.returncode = -signal.SIGINT
        elif signum == signal.SIGTERM:
            self.terminated = True
            self.returncode = -signal.SIGTERM
        elif signum == signal.SIGKILL:
            self.killed = True
            self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        if self.returncode is None:
            if _FakePopen.wait_completes:
                self.returncode = 0
                return 0
            raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout)
        return self.returncode


@pytest.fixture()
def fake_popen(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _FakePopen.instances = []
    _FakePopen.wait_completes = False
    _FakePopen.interrupt_exits = True
    monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        launcher, "_signal_process_group", lambda child, signum: child.send_signal(signum)
    )
    monkeypatch.setattr(launcher, "_process_group_exists", lambda child: child.returncode is None)
    monkeypatch.setattr(launcher, "_COOPERATIVE_EXIT_GRACE_S", 0.0)
    monkeypatch.setattr(launcher, "_TERMINATE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(launcher, "_NORMAL_EXIT_GRACE_S", 0.01)
    return _FakePopen


# ---------------------------------------------------------------------------
# launch_torchrun_workers
# ---------------------------------------------------------------------------


def test_launch_torchrun_workers_builds_standard_single_node_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    def fake_run(command, *, env, check):
        captured.update(command=list(command), env=dict(env), check=check)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    script = tmp_path / "train_rsl_rl.py"

    launch_torchrun_workers(
        world_size=2,
        script_path=script,
        argv=["task=go2_joystick_flat/mujoco"],
        log_dir="/tmp/ppo_gpux2",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert "--nnodes=1" in command
    assert "--nproc_per_node=2" in command
    assert command[-1:] == ["task=go2_joystick_flat/mujoco"]
    launch_env = captured["env"]
    assert isinstance(launch_env, dict)
    assert launch_env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert launch_env["NCCL_P2P_DISABLE"] == "1"
    assert launch_env["NCCL_SHM_DISABLE"] == "1"
    assert launch_env[UNILAB_DP_LOG_DIR] == "/tmp/ppo_gpux2"
    assert captured["check"] is False


def test_launch_torchrun_workers_propagates_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 7})(),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        launch_torchrun_workers(
            world_size=2,
            script_path="scripts/train_rsl_rl.py",
            argv=[],
            log_dir="/tmp/ppo_gpux2",
        )


def test_launch_torchrun_workers_preserves_explicit_nccl_transport(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_run(*args, env, **kwargs):
        del args, kwargs
        captured.update(env)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    monkeypatch.setenv("NCCL_SHM_DISABLE", "0")

    launch_torchrun_workers(
        world_size=2,
        script_path="scripts/train_rsl_rl.py",
        argv=[],
        log_dir="/tmp/ppo_gpux2",
    )

    assert captured["NCCL_P2P_DISABLE"] == "0"
    assert captured["NCCL_SHM_DISABLE"] == "0"


# ---------------------------------------------------------------------------
# DpRankSupervisor
# ---------------------------------------------------------------------------


def test_single_device_topology_spawns_no_children(fake_popen, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-rank-local")
    with DpRankSupervisor(world_size=1, log_dir="/tmp/dp_test_log"):
        assert fake_popen.instances == []
    assert os.environ.get(UNILAB_DP_RANK) is None


def test_supervisor_spawn_argv_and_env(fake_popen, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(sys, "argv", ["train_sac.py", "task=g1_walk_flat/mujoco"])
    with DpRankSupervisor(world_size=3, log_dir="/tmp/dp_test_log"):
        assert len(fake_popen.instances) == 2
        for rank, child in enumerate(fake_popen.instances, start=1):
            assert child.argv[0] == sys.executable
            assert child.argv[1] == sys.argv[0]
            assert child.argv[2:] == ["task=g1_walk_flat/mujoco"]
            assert child.start_new_session is (os.name == "posix")
            assert child.env[UNILAB_DP_RANK] == str(rank)
            assert child.env[UNILAB_DP_WORLD_SIZE] == "3"
            assert child.env["CUDA_VISIBLE_DEVICES"] == str(rank)
            assert child.env[UNILAB_DP_LOG_DIR] == "/tmp/dp_test_log"
        # Rank 0's own environment stays untouched.
        assert os.environ.get(UNILAB_DP_RANK) is None
        for child in fake_popen.instances:
            child.returncode = 0
    for child in fake_popen.instances:
        assert not child.terminated


def test_supervisor_children_each_receive_one_visible_gpu(
    fake_popen, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Visibility tokens pass through to spawned ranks opaquely."""
    monkeypatch.delenv(UNILAB_DP_RANK, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")

    with DpRankSupervisor(world_size=3, log_dir="/tmp/dp_test_log"):
        for child in fake_popen.instances:
            child.returncode = 0

    assert len(fake_popen.instances) == 2
    assert [child.env["CUDA_VISIBLE_DEVICES"] for child in fake_popen.instances] == [
        "GPU-b",
        "GPU-c",
    ]


def test_supervisor_normal_exit_waits_for_children(fake_popen):
    _FakePopen.wait_completes = True
    with DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log"):
        pass
    child = fake_popen.instances[0]
    assert child.returncode == 0
    assert not child.terminated


def test_supervisor_grace_timeout_is_a_failure(fake_popen):
    with pytest.raises(RuntimeError, match="rank 1 exit code timeout"):
        with DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log"):
            pass
    child = fake_popen.instances[0]
    assert child.terminated
    assert child.returncode == -signal.SIGTERM


def test_supervisor_clean_child_exit_is_not_a_failure(fake_popen):
    with DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log"):
        fake_popen.instances[0].returncode = 0


def test_supervisor_failed_child_makes_rank_zero_fail(fake_popen, monkeypatch: pytest.MonkeyPatch):
    # Keep the watchdog from polling so __exit__ observes the exit code first.
    monkeypatch.setattr(launcher, "_WATCHDOG_INTERVAL_S", 60.0)
    supervisor = DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log")
    supervisor.__enter__()
    fake_popen.instances[0].returncode = 3
    with pytest.raises(RuntimeError, match="rank 1 exit code 3"):
        supervisor.__exit__(None, None, None)


def test_supervisor_error_exit_terminates_live_children(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    fake_popen.interrupt_exits = False
    with pytest.raises(ValueError, match="boom"):
        with DpRankSupervisor(world_size=3, log_dir="/tmp/dp_test_log"):
            raise ValueError("boom")
    assert all(child.terminated for child in fake_popen.instances)
    assert all(child.returncode == -signal.SIGTERM for child in fake_popen.instances)
    assert all(child.signals == [signal.SIGINT, signal.SIGTERM] for child in fake_popen.instances)


def test_supervisor_ctrl_c_forwards_sigint_for_cooperative_rank_cleanup(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    previous_sigint = signal.getsignal(signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        with DpRankSupervisor(world_size=3, log_dir="/tmp/dp_test_log") as supervisor:
            installed = signal.getsignal(signal.SIGINT)
            assert getattr(installed, "__self__", None) is supervisor
            installed(signal.SIGINT, None)

    assert all(child.signals == [signal.SIGINT] for child in fake_popen.instances)
    assert all(child.returncode == -signal.SIGINT for child in fake_popen.instances)
    assert signal.getsignal(signal.SIGINT) == previous_sigint


def test_supervisor_watchdog_sigterms_rank_zero_on_child_death(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(launcher, "_WATCHDOG_INTERVAL_S", 0.01)
    killed: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: killed.append(sig))
    with pytest.raises(RuntimeError, match="exit code 1"):
        with DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log"):
            fake_popen.instances[0].returncode = 1
            deadline = time.monotonic() + 5.0
            while not killed and time.monotonic() < deadline:
                time.sleep(0.01)
    assert killed == [signal.SIGTERM]


def test_supervisor_restores_signal_handlers(fake_popen):
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    with DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log") as supervisor:
        assert getattr(signal.getsignal(signal.SIGINT), "__self__", None) is supervisor
        assert getattr(signal.getsignal(signal.SIGTERM), "__self__", None) is supervisor
        fake_popen.instances[0].returncode = 0
    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


def test_supervisor_restores_signal_handlers_when_group_reaping_fails(
    fake_popen, monkeypatch: pytest.MonkeyPatch
):
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    supervisor = DpRankSupervisor(world_size=2, log_dir="/tmp/dp_test_log")
    supervisor.__enter__()
    fake_popen.instances[0].returncode = 0

    def fail_reaping() -> None:
        raise RuntimeError("group reaping failed")

    monkeypatch.setattr(supervisor, "_terminate_children", fail_reaping)

    with pytest.raises(RuntimeError, match="group reaping failed"):
        supervisor.__exit__(None, None, None)

    assert signal.getsignal(signal.SIGINT) == previous_sigint
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm
