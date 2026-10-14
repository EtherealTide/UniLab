"""Controller tests for the owner-owned env-update graph runtime."""

from __future__ import annotations

from typing import Any

import pytest
from unisim import GraphCompletionToken, GraphLifecycleState

from unilab.base.env_update_graph import validate_env_update_graph_manifest
from tests.envs.test_manager_based_rl_env import _make_cfg, _make_env


class FakeEnvUpdateGraphRuntime:
    """Bounded runtime double; production CUDA capture lands in a later phase."""

    def __init__(self, *, fail_capture: bool = False, fail_replay: bool = False):
        self.stream_id = 17
        self.layout: dict[str, Any] | None = None
        self.capture_id = "env-update-capture"
        self.capture_calls = 0
        self.replay_calls = 0
        self.fail_capture = fail_capture
        self.fail_replay = fail_replay

    def capture(self, verdict, body, *, current_layout):
        del verdict
        if self.fail_capture:
            raise RuntimeError("simulated capture failure")
        self.capture_calls += 1
        body()
        self.layout = current_layout
        return self.capture_id, current_layout

    def replay(self, receipt, body):
        del receipt, body
        if self.fail_replay:
            self.fail_replay = False
            raise RuntimeError("simulated replay failure")
        self.replay_calls += 1
        return GraphCompletionToken(
            contract_version=1,
            lifecycle_state=GraphLifecycleState.READY,
            operation="env_update",
            sequence=self.replay_calls,
            stream_id=self.stream_id,
            capture_id=self.capture_id,
        )


def _make_graph_env():
    cfg = _make_cfg(include_optional_managers=False)
    cfg.events = {}
    env, backend = _make_env(cfg, torch_stream_id=17)
    env.init_state()
    env.set_deferred_validation_boundaries(True)
    return env, backend


def test_env_update_runtime_rejects_cpu_before_capture() -> None:
    env, _backend = _make_graph_env()
    runtime = FakeEnvUpdateGraphRuntime()
    env._env_update_graph_runtime = runtime

    with pytest.raises(RuntimeError, match="environment carriers must be CUDA-resident"):
        env.configure_env_update_graph("graph")
    assert runtime.capture_calls == 0
    manifest = env.get_env_update_graph_manifest()
    assert manifest["enabled"] is False
    validate_env_update_graph_manifest(manifest)
    env.close()


def test_env_update_graph_requires_runtime_before_capture() -> None:
    env, _backend = _make_graph_env()
    # Runtime absence is checked after the owner's fail-closed carrier checks.
    with pytest.raises(RuntimeError, match="CUDA-resident"):
        env.configure_env_update_graph("graph")
    env.close()


def test_env_update_runtime_requires_backend_receipt_for_replay() -> None:
    env, _backend = _make_graph_env()
    runtime = FakeEnvUpdateGraphRuntime()
    env._env_update_graph_runtime = runtime
    with pytest.raises(RuntimeError, match="requires the backend step receipt"):
        env._replay_env_update_graph_body(env._state.info["steps"])
    env.close()
