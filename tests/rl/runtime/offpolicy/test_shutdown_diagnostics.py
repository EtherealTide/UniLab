"""Tests for host-only off-policy shutdown diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from unilab.rl.runtime.offpolicy.shutdown_diagnostics import ShutdownDiagnosticsRecorder


def test_shutdown_recorder_reports_phase_ring_replay_and_collector() -> None:
    recorder = ShutdownDiagnosticsRecorder()
    recorder.set_phase(
        owner="learner",
        phase="training/learner_update",
        iteration=7,
        coordination_tick=6,
    )
    recorder.record_failure(RuntimeError("learner update failed"))
    recorder.record_cleanup_error(OSError("collector join failed"))
    coordination = SimpleNamespace(snapshot=lambda: ("busy", 12))
    ring = SimpleNamespace(
        diagnostics=lambda: {"epoch": 0, "published_tick": 6, "consumed_tick": 6}
    )
    replay = SimpleNamespace(ingress_diagnostics=lambda: {"occupancy": 0})
    collector = SimpleNamespace(is_alive=lambda: False, exitcode=-9)

    snapshot = recorder.snapshot(
        learner_coordination=coordination,
        inference_ring=ring,
        replay_buffer=replay,
        collector_process=collector,
    )

    assert snapshot["classification"] == "learner_failure"
    assert snapshot["owner"] == "learner"
    assert snapshot["phase"] == "training/learner_update"
    assert snapshot["iteration"] == 7
    assert snapshot["coordination_tick"] == 6
    assert snapshot["learner_coordination"] == {"phase": "busy", "progress": 12}
    assert snapshot["inference_ring"]["published_tick"] == 6
    assert snapshot["replay_ingress"]["occupancy"] == 0
    assert snapshot["collector"] == {
        "alive": False,
        "exitcode": -9,
        "signal": 9,
        "signal_name": "SIGKILL",
    }
    assert snapshot["cleanup"]["errors"] == [
        {"type": "OSError", "message": "collector join failed"}
    ]


def test_shutdown_recorder_classifies_external_cancellation() -> None:
    recorder = ShutdownDiagnosticsRecorder()
    recorder.record_failure(KeyboardInterrupt())

    snapshot = recorder.snapshot()

    assert snapshot["classification"] == "external_cancellation"
    assert snapshot["owner"] == "external"
    assert snapshot["exception"]["type"] == "KeyboardInterrupt"


def test_shutdown_recorder_snapshots_are_best_effort() -> None:
    class Failing:
        def snapshot(self):
            raise RuntimeError("coordination unavailable")

        def diagnostics(self):
            raise RuntimeError("ring unavailable")

        def ingress_diagnostics(self):
            raise RuntimeError("replay unavailable")

        def is_alive(self):
            raise RuntimeError("collector unavailable")

    recorder = ShutdownDiagnosticsRecorder()
    failing = Failing()

    snapshot = recorder.snapshot(
        learner_coordination=failing,
        inference_ring=failing,
        replay_buffer=failing,
        collector_process=failing,
    )

    assert snapshot["learner_coordination"] == {"phase": None, "progress": None}
    assert snapshot["inference_ring"] is None
    assert snapshot["replay_ingress"] is None
    assert snapshot["collector"] is None


def test_shutdown_recorder_reset_clears_previous_run_and_keeps_epoch() -> None:
    recorder = ShutdownDiagnosticsRecorder(inference_epoch=3)
    recorder.set_phase(owner="learner", phase="training/learner_update", iteration=4)
    recorder.record_failure(RuntimeError("previous run"))
    recorder.record_cleanup_error(OSError("previous cleanup"))

    recorder.reset(inference_epoch=5)

    assert recorder.owner == "learner"
    assert recorder.phase == "startup/begin"
    assert recorder.iteration is None
    assert recorder.coordination_tick is None
    assert recorder.inference_epoch == 5
    assert recorder.classification == "unknown_failure"
    assert recorder.exception_type is None
    assert recorder.exception_message is None
    assert recorder.cleanup_errors == []


def test_shutdown_recorder_contains_no_nested_schema_version() -> None:
    snapshot = ShutdownDiagnosticsRecorder().snapshot()

    assert "schema_version" not in snapshot
