"""Parent-death watchdog for collector subprocesses.

``multiprocessing.Process(daemon=True)`` only stops a child when the parent
exits through the Python interpreter. A learner killed by ``SIGKILL`` (OOM
killer, external kill, test-timeout kill) never runs that path, so spawn
collectors are reparented to PID 1 and keep running the env hot loop on a
``stop_event`` nobody will ever set — burning CPU and pinning shared-memory
and semaphore resources.

The watchdog is a daemon thread installed at collector entry. It records the
parent PID at spawn time and polls ``os.getppid()``: the kernel reparents the
collector only when the original parent is gone, so a changed ppid is an
unambiguous death signal. On detection the watchdog

1. emits a diagnostic on stderr,
2. sets the shared ``stop_event`` so the collector drains gracefully, and
3. after a bounded grace period forces ``os._exit`` — a raw syscall that
   also fires when interpreter finalization is wedged joining non-daemon
   native threads, since this daemon thread is still running at that stage.

The parent PID is handed from the spawner to the child at launch. Reading
``os.getppid()`` inside the child is not a reliable substitute: a parent
killed during the child's slow spawn boot leaves the child already
reparented to PID 1 before its entry point runs, and a watchdog that
records ppid=1 as its parent would never fire.

The mechanism is POSIX-portable (macOS and Linux) and deliberately does not
rely on ``daemon=True`` or platform-specific facilities such as Linux
``prctl(PR_SET_PDEATHSIG)`` — the ppid poll already closes the parent-death
race those syscalls leave open.
"""

from __future__ import annotations

import os
import threading
import time

DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_GRACE_PERIOD_S = 10.0
FORCE_EXIT_CODE = 1


def _emit(message: str) -> None:
    """Best-effort diagnostic that survives interpreter finalization."""
    try:
        os.write(2, f"{message}\n".encode())
    except OSError:
        pass


def parent_is_gone(parent_pid: int) -> bool:
    """Return whether the recorded parent PID is no longer our parent.

    Reparenting (to PID 1 or a subreaper) is the only way ``getppid``
    changes, and the kernel only reparents once the original parent has
    exited, so this needs no pid-reuse guard.
    """
    return os.getppid() != parent_pid


def install_parent_watchdog(
    stop_event=None,
    *,
    parent_pid: int | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_S,
    grace_period: float = DEFAULT_GRACE_PERIOD_S,
    label: str = "collector",
) -> threading.Thread | None:
    """Install the parent-death watchdog for the current (collector) process.

    ``parent_pid`` must be passed by the spawner (its own ``os.getpid()``):
    the child cannot reliably learn it via ``os.getppid()`` because a parent
    killed during the child's slow spawn boot leaves the child already
    reparented to PID 1 before this code runs. ``None`` falls back to
    ``os.getppid()`` at call time, which is only safe when the parent is
    known to be alive. Returns the started daemon thread.

    Returns ``None`` without installing anything when ``parent_pid`` is the
    current process: the entry point is then running inline in the
    spawner's own process (test doubles, not a real spawn child), and a
    watchdog would misdetect "parent death" on its first poll and
    force-exit the host process once the grace period expires.
    """
    if parent_pid is None:
        parent_pid = os.getppid()
    if parent_pid == os.getpid():
        return None
    thread = threading.Thread(
        target=_watch_loop,
        args=(parent_pid, stop_event, poll_interval, grace_period, label),
        name=f"{label}-parent-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _watch_loop(
    parent_pid: int,
    stop_event,
    poll_interval: float,
    grace_period: float,
    label: str,
) -> None:
    while not parent_is_gone(parent_pid):
        time.sleep(poll_interval)

    _emit(
        f"[{label}] parent process {parent_pid} is gone "
        f"(reparented to pid {os.getppid()}); requesting stop, "
        f"force-exit in {grace_period:.1f}s if still alive"
    )
    if stop_event is not None:
        try:
            stop_event.set()
        except Exception:
            pass

    deadline = time.monotonic() + grace_period
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    _emit(
        f"[{label}] graceful shutdown did not finish within "
        f"{grace_period:.1f}s of parent death; forcing exit"
    )
    os._exit(FORCE_EXIT_CODE)


__all__ = [
    "DEFAULT_GRACE_PERIOD_S",
    "DEFAULT_POLL_INTERVAL_S",
    "FORCE_EXIT_CODE",
    "install_parent_watchdog",
    "parent_is_gone",
]
