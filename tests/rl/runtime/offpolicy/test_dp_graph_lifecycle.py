"""Captured collectives must release their communicator before group teardown."""

from types import SimpleNamespace

import pytest

from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.rl.runtime.offpolicy.runner import OffPolicyRunner


@pytest.mark.parametrize("ipc_fails", [False, True])
def test_runner_releases_captured_communication_before_process_group(monkeypatch, ipc_fails):
    events = []

    def close_ipc(_runner):
        events.append("ipc")
        if ipc_fails:
            raise RuntimeError("ipc cleanup failed")

    def detach(sync):
        assert sync is None
        events.append("graph")

    monkeypatch.setattr(OffPolicyRunner, "close", close_ipc)
    runner = object.__new__(DoubleBufferOffPolicyRunner)
    runner.learner = SimpleNamespace(set_gradient_sync=detach)
    runner.dp_sync = SimpleNamespace(close=lambda: events.append("group"))
    if ipc_fails:
        with pytest.raises(RuntimeError, match="ipc cleanup failed"):
            runner.close()
    else:
        runner.close()
    assert events == ["ipc", "graph", "group"]
