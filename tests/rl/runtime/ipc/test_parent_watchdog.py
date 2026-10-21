"""Tests for the collector parent-death watchdog.

Covers the three collector lifecycle paths:

- parent alive: the watchdog never fires and stop_event drains cleanly;
- learner SIGKILL: an orphaned collector stops within the bounded
  poll + grace window instead of running the env hot loop forever;
- wedged collector: a child that never observes stop_event is force-exited
  after the grace period (the finalization-hang backstop).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time

from unilab.rl.runtime.async_runner import _collector_entry_wrapper
from unilab.rl.runtime.ipc.parent_watchdog import (
    FORCE_EXIT_CODE,
    install_parent_watchdog,
    parent_is_gone,
)

_SPAWN_CTX = mp.get_context("spawn")

_POLL = 0.1
_GRACE = 2.0
# Generous CI slack; the expected bound is poll + grace (~2.1s).
_DEATH_TIMEOUT = 15.0


def _stop_aware_child(stop_event) -> None:
    """Collector stub that honors stop_event like the real hot loops."""
    while not stop_event.is_set():
        time.sleep(0.05)


def _stop_deaf_child(stop_event) -> None:
    """Collector stub stuck in native work that never observes stop_event."""
    del stop_event
    while True:
        time.sleep(0.05)


def _orphaned_collector_harness(child_kind: str, pid_queue) -> None:
    """Fake learner: spawns a collector through the real entry wrapper.

    The test SIGKILLs this process, which orphans the collector exactly
    like a killed learner does (daemon=True cannot help here).
    """
    stop_event = _SPAWN_CTX.Event()
    target = {"aware": _stop_aware_child, "deaf": _stop_deaf_child}[child_kind]
    proc = _SPAWN_CTX.Process(
        target=_collector_entry_wrapper,
        args=(
            target,
            None,
            {
                "stop_event": stop_event,
                "_parent_pid": os.getpid(),
                "_parent_watchdog_poll_interval": _POLL,
                "_parent_watchdog_grace_period": _GRACE,
            },
        ),
        daemon=True,
    )
    proc.start()
    pid_queue.put(proc.pid)
    while True:
        time.sleep(1.0)


def _pid_dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Exists but owned by another uid; treat as alive.
        return False
    # Linux: after reparenting, a container PID 1 may not reap the exited
    # collector promptly; a zombie still answers kill(pid, 0).
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rpartition(")")[2].split()[0] == "Z"
    except FileNotFoundError:
        # No /proc (macOS): the kill above proves the process is alive.
        # With /proc present, the process exited between the two checks.
        return os.path.exists("/proc")


def _wait_pid_dead(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_dead(pid):
            return True
        time.sleep(0.1)
    return False


def _spawn_orphaned_collector(child_kind: str) -> int:
    """Run the fake learner, SIGKILL it, and return the orphaned child pid."""
    pid_queue = _SPAWN_CTX.Queue()
    learner = _SPAWN_CTX.Process(
        target=_orphaned_collector_harness,
        args=(child_kind, pid_queue),
        # Non-daemon like the real learner: daemonic processes cannot have
        # children, and the test kills it with SIGKILL anyway.
        daemon=False,
    )
    learner.start()
    try:
        assert learner.pid is not None
        child_pid = pid_queue.get(timeout=30)
        os.kill(learner.pid, signal.SIGKILL)
        learner.join(timeout=10)
        assert not learner.is_alive()
        return child_pid
    finally:
        if learner.is_alive():
            learner.kill()
            learner.join(timeout=10)
        pid_queue.close()
        pid_queue.join_thread()


class TestParentIsGone:
    def test_current_parent_is_alive(self):
        assert parent_is_gone(os.getppid()) is False

    def test_any_other_pid_counts_as_reparented(self):
        assert parent_is_gone(os.getppid() + 1) is True


class TestInstallParentWatchdog:
    def test_returns_live_daemon_thread(self):
        thread = install_parent_watchdog(poll_interval=60.0, grace_period=60.0)
        assert thread.daemon
        assert thread.is_alive()
        assert "parent-watchdog" in thread.name

    def test_self_pid_installs_no_watchdog(self):
        # parent_pid == own pid means the entry point runs inline in the
        # spawner's process (test double); a watchdog there would force-exit
        # the host after the grace period.
        assert install_parent_watchdog(parent_pid=os.getpid()) is None

    def test_inline_entry_wrapper_installs_no_watchdog(self):
        # Regression: _InlineProcess-style test doubles run the entry
        # wrapper in-process; this must not start a watchdog that would
        # os._exit the host test runner once the grace period expires.
        stop = threading.Event()
        stop.set()
        before = {t.ident for t in threading.enumerate()}
        _collector_entry_wrapper(
            _stop_aware_child,
            None,
            {"stop_event": stop, "_parent_pid": os.getpid()},
        )
        new_threads = [t for t in threading.enumerate() if t.ident not in before]
        assert not any("parent-watchdog" in t.name for t in new_threads)


class TestParentAlive:
    def test_watchdog_never_fires_and_stop_event_drains_cleanly(self):
        stop_event = _SPAWN_CTX.Event()
        proc = _SPAWN_CTX.Process(
            target=_collector_entry_wrapper,
            args=(
                _stop_aware_child,
                None,
                {
                    "stop_event": stop_event,
                    "_parent_pid": os.getpid(),
                    "_parent_watchdog_poll_interval": _POLL,
                    "_parent_watchdog_grace_period": _GRACE,
                },
            ),
            daemon=True,
        )
        proc.start()
        try:
            time.sleep(10 * _POLL)
            assert proc.is_alive(), "watchdog fired while the parent is alive"
            stop_event.set()
            proc.join(timeout=10)
            assert proc.exitcode == 0
        finally:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)


class TestLearnerSigkill:
    def test_orphaned_collector_stops_via_stop_event(self):
        child_pid = _spawn_orphaned_collector("aware")
        assert _wait_pid_dead(child_pid, _DEATH_TIMEOUT), (
            f"orphaned collector {child_pid} still alive {_DEATH_TIMEOUT}s after learner SIGKILL"
        )

    def test_wedged_collector_is_force_exited_after_grace(self):
        child_pid = _spawn_orphaned_collector("deaf")
        # The deaf child ignores stop_event; only the os._exit backstop
        # after the grace period can reap it.
        time.sleep(_POLL + 0.2)
        assert _wait_pid_dead(child_pid, _DEATH_TIMEOUT), (
            f"wedged collector {child_pid} survived the watchdog grace period"
        )


def test_force_exit_code_is_nonzero_and_signal_free():
    # Parent-side diagnostics treat signal exits and code 0 specially; the
    # watchdog backstop must be distinguishable from both.
    assert FORCE_EXIT_CODE not in (0, -15)
