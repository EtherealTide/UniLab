"""Host-only shutdown diagnostics for the double-buffer off-policy runner."""

from __future__ import annotations

import signal
from collections.abc import Callable
from typing import Any

_TRACEBACK_MAX_CHARS = 16_000


class ShutdownDiagnosticsRecorder:
    """Track the last safe lifecycle phase without touching accelerator tensors.

    The recorder is intentionally host-only.  Every optional snapshot is
    best-effort because diagnostics must never replace the training exception.
    """

    def __init__(self, inference_epoch: int | None = None) -> None:
        self.reset(inference_epoch=inference_epoch)

    def reset(self, inference_epoch: int | None = None) -> None:
        self.owner = "learner"
        self.phase = "startup/begin"
        self.iteration: int | None = None
        self.coordination_tick: int | None = None
        self.inference_epoch = None if inference_epoch is None else int(inference_epoch)
        self.classification: str = "unknown_failure"
        self.exception_type: str | None = None
        self.exception_message: str | None = None
        self.cleanup_errors: list[dict[str, str]] = []

    def set_phase(
        self,
        *,
        owner: str,
        phase: str,
        iteration: int | None = None,
        coordination_tick: int | None = None,
    ) -> None:
        self.owner = owner
        self.phase = phase
        if iteration is not None:
            self.iteration = int(iteration)
        if coordination_tick is not None:
            self.coordination_tick = int(coordination_tick)

    def record_inference_epoch(self, epoch: int) -> None:
        self.inference_epoch = int(epoch)

    def record_normal_completion(self) -> None:
        self.classification = "normal_completion"
        self.exception_type = None
        self.exception_message = None

    def record_failure(self, exc: BaseException, *, owner: str = "learner") -> None:
        self.classification = (
            "external_cancellation"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "learner_failure"
        )
        self.owner = "external" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else owner
        self.exception_type = type(exc).__name__
        self.exception_message = str(exc)[:_TRACEBACK_MAX_CHARS]

    def record_collector_failure(self, exc: BaseException) -> None:
        self.record_failure(exc, owner="collector")
        self.classification = "collector_failure"

    def record_cleanup_error(self, exc: BaseException) -> None:
        self.cleanup_errors.append(
            {"type": type(exc).__name__, "message": str(exc)[:_TRACEBACK_MAX_CHARS]}
        )

    def snapshot(
        self,
        *,
        learner_coordination=None,
        inference_ring=None,
        replay_buffer=None,
        collector_process=None,
    ) -> dict[str, object]:
        coordination = _safe_snapshot(learner_coordination, "snapshot")
        ring = _safe_snapshot(inference_ring, "diagnostics")
        replay_ingress = _safe_snapshot(replay_buffer, "ingress_diagnostics")
        collector = _collector_snapshot(collector_process)
        phase, progress = coordination if isinstance(coordination, tuple) else (None, None)
        return {
            "classification": self.classification,
            "owner": self.owner,
            "phase": self.phase,
            "iteration": self.iteration,
            "coordination_tick": self.coordination_tick,
            "inference_epoch": self.inference_epoch,
            "exception": self._exception_snapshot(),
            "learner_coordination": {
                "phase": getattr(phase, "name", phase),
                "progress": progress,
            },
            "inference_ring": ring,
            "replay_ingress": replay_ingress,
            "collector": collector,
            "cleanup": {"errors": list(self.cleanup_errors)},
        }

    def _exception_snapshot(self) -> dict[str, str | None] | None:
        if self.exception_type is None and self.exception_message is None:
            return None
        return {"type": self.exception_type, "message": self.exception_message}


def _safe_snapshot(source: Any, method_name: str) -> Any:
    method: Callable[[], Any] | None = getattr(source, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except BaseException:
        return None


def _collector_snapshot(process: Any) -> dict[str, object] | None:
    if process is None:
        return None
    try:
        alive = bool(process.is_alive())
        exitcode = getattr(process, "exitcode", None)
    except BaseException:
        return None
    signal_number = -exitcode if isinstance(exitcode, int) and exitcode < 0 else None
    signal_name = None
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = None
    return {
        "alive": alive,
        "exitcode": exitcode,
        "signal": signal_number,
        "signal_name": signal_name,
    }


__all__ = ["ShutdownDiagnosticsRecorder"]
