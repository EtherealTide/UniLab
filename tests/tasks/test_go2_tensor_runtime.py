from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    SelectedResetPublication,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.tasks.locomotion.go2 import tensor_parity as tensor_parity_module
from unilab.tasks.locomotion.go2.tensor_parity import (
    GO2_ACTUATOR_NAMES,
    GO2_JOINT_NAMES,
    Go2TensorParityConfig,
    Go2TensorParityRuntime,
)

_HOME = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float32,
)
_JOINT_TO_ACTUATOR = tuple(
    GO2_ACTUATOR_NAMES.index(name.removesuffix("_joint")) for name in GO2_JOINT_NAMES
)


class FakeGo2DeviceBackend:
    """SDK-free fake exercising the public tensor contract, not a real backend."""

    backend_type = "fake-go2-device"

    def __init__(self, num_envs: int = 3):
        self.num_envs = int(num_envs)
        self.num_actuators = 12
        self.qpos = torch.zeros((self.num_envs, 19), dtype=torch.float32)
        self.qpos[:, 2] = 0.3
        self.qpos[:, 3] = 1.0
        self.qpos[:, 7:] = torch.tensor(_HOME)
        self.qvel = torch.zeros((self.num_envs, 18), dtype=torch.float32)
        self.ctrl = torch.zeros((self.num_envs, 12), dtype=torch.float32)
        self.sensors = torch.zeros((self.num_envs, 25), dtype=torch.float32)
        self.state_negotiations = 0
        self.sensor_negotiations = 0
        self.step_calls = 0
        self.reset_calls = 0
        self.cpu_detour_calls = 0
        self._refresh_sensors()

    def tensor_execution(self) -> TensorExecution:
        return TensorExecution.DEVICE_RESIDENT

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=True,
            stepping=True,
            selected_reset=True,
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            mask_selected_reset=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="backend-owned synchronous current stream",
            torch_devices=("cpu",),
        )

    def get_state_views(self, fields, device: str | torch.device = "cpu"):
        assert tuple(fields or ()) == ("qpos", "qvel")
        assert torch.device(device).type == "cpu"
        self.state_negotiations += 1
        return {"qpos": self.qpos, "qvel": self.qvel}

    def get_sensor_view(self, name: str, device: str | torch.device = "cpu"):
        assert torch.device(device).type == "cpu"
        self.sensor_negotiations += 1
        layout = {
            "gyro": (0, 3),
            "local_linvel": (3, 3),
            "upvector": (6, 3),
            "FL_foot_contact": (9, 1),
            "FR_foot_contact": (10, 1),
            "RL_foot_contact": (11, 1),
            "RR_foot_contact": (12, 1),
            "FL_pos": (13, 3),
            "FR_pos": (16, 3),
            "RL_pos": (19, 3),
            "RR_pos": (22, 3),
        }
        if name not in layout:
            raise KeyError(name)
        start, width = layout[name]
        return self.sensors[:, start : start + width]

    def step_tensor(self, ctrl, nsteps: int = 1):
        assert isinstance(ctrl, torch.Tensor)
        assert ctrl.shape == self.ctrl.shape
        assert ctrl.dtype is torch.float32
        assert bool(torch.isfinite(ctrl).all())
        self.ctrl.copy_(ctrl)
        self.qpos[:, 2] += 0.01 * nsteps
        self.qpos[:, 7:] += 0.02 * nsteps
        self.qvel[:, :3] += 0.01 * nsteps
        self.qvel[:, 3:6] -= 0.02 * nsteps
        self._refresh_sensors()
        self.step_calls += 1
        return {"ok": True}

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None):
        assert randomization is None
        assert reset_mask.dtype is torch.bool
        assert reset_mask.shape == (self.num_envs,)
        assert bool(torch.isfinite(qpos).all())
        assert bool(torch.isfinite(qvel).all())
        if not bool(reset_mask.any()):
            return None
        self.qpos[reset_mask] = qpos[reset_mask]
        self.qvel[reset_mask] = qvel[reset_mask]
        self._refresh_sensors(reset_mask)
        self.reset_calls += 1
        return {"ok": True}

    def _refresh_sensors(self, reset_mask: torch.Tensor | None = None) -> None:
        selected = torch.ones(self.num_envs, dtype=torch.bool) if reset_mask is None else reset_mask
        if not bool(selected.any()):
            return
        w, x, y, z = self.qpos[selected, 3:7].unbind(dim=1)
        self.sensors[selected, 0:3] = self.qvel[selected, 3:6]
        self.sensors[selected, 3:6] = self.qvel[selected, 0:3]
        self.sensors[selected, 6] = 2.0 * (x * z + w * y)
        self.sensors[selected, 7] = 2.0 * (y * z - w * x)
        self.sensors[selected, 8] = 1.0 - 2.0 * (x * x + y * y)
        contact = (self.qpos[selected][:, [8, 11, 14, 17]] > 0.1).to(torch.float32)
        self.sensors[selected, 9:13] = contact
        self.sensors[selected, 13] = self.qpos[selected, 7]
        self.sensors[selected, 16] = self.qpos[selected, 10]
        self.sensors[selected, 19] = self.qpos[selected, 13]
        self.sensors[selected, 22] = self.qpos[selected, 16]


class UnsupportedBackend:
    backend_type = "fake-unsupported"

    def tensor_execution(self) -> TensorExecution:
        return TensorExecution.UNSUPPORTED

    def get_tensor_capabilities(self) -> TensorLifecycleCapabilities:
        return TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)


def _reset_state(num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = torch.zeros((num_envs, 19), dtype=torch.float32)
    qpos[:, 2] = 0.3
    qpos[:, 3] = 1.0
    qpos[:, 7:] = torch.tensor(_HOME)
    qvel = torch.zeros((num_envs, 18), dtype=torch.float32)
    return qpos, qvel


def _make_runtime(
    cfg: Go2TensorParityConfig | None = None,
) -> tuple[FakeGo2DeviceBackend, Go2TensorParityRuntime]:
    backend = FakeGo2DeviceBackend()
    qpos, qvel = _reset_state(backend.num_envs)
    command = torch.tensor(
        [[0.4, -0.2, 0.3], [-0.3, 0.1, -0.2], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    runtime = Go2TensorParityRuntime(
        backend,  # pyright: ignore[reportArgumentType]
        command=command,
        reset_qpos=qpos,
        reset_qvel=qvel,
        cfg=cfg or Go2TensorParityConfig(),
    )
    runtime.reset()
    return backend, runtime


def test_go2_tensor_runtime_rejects_unsupported_backend_before_view_negotiation() -> None:
    qpos, qvel = _reset_state(1)
    with pytest.raises(RuntimeError, match="DEVICE_RESIDENT"):
        Go2TensorParityRuntime(
            UnsupportedBackend(),  # pyright: ignore[reportArgumentType]
            command=torch.zeros((1, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
        )


def test_go2_full_tensor_lifecycle_matches_owner_formula_parity() -> None:
    backend, runtime = _make_runtime()
    actions = torch.tensor(
        [
            [0.1] * 12,
            [-0.2] * 12,
            [0.3] * 12,
        ],
        dtype=torch.float32,
    )
    state = runtime.step(actions)

    assert state.obs["policy"].shape == (3, 49)
    assert state.obs["critic"].shape == (3, 52)
    assert backend.step_calls == 1
    assert backend.state_negotiations == 1
    assert backend.sensor_negotiations == len(runtime.sensors)
    assert backend.cpu_detour_calls == 0

    expected_ctrl = torch.from_numpy(np.asarray(_HOME)[list(_JOINT_TO_ACTUATOR)]).repeat(3, 1)
    expected_ctrl += 0.25 * actions
    torch.testing.assert_close(backend.ctrl, expected_ctrl)

    gyro = backend.qvel[:, 3:6]
    linvel = backend.qvel[:, 0:3]
    upvector = backend.sensors[:, 6:9]
    joint_pos = backend.qpos[:, 7:]
    command = runtime.command
    expected_policy = torch.cat(
        (
            gyro,
            -upvector,
            joint_pos - torch.tensor(_HOME),
            backend.qvel[:, 6:],
            actions,
            command,
            torch.tensor([0.04, 0.54, 0.54, 0.04]).repeat(3, 1),
        ),
        dim=1,
    )
    torch.testing.assert_close(state.obs["policy"], expected_policy)
    torch.testing.assert_close(state.obs["critic"], torch.cat((expected_policy, linvel), dim=1))

    expected = {
        "tracking_lin_vel": torch.exp(
            -torch.sum((command[:, :2] - linvel[:, :2]) ** 2, dim=1) / 0.25
        ),
        "tracking_ang_vel": torch.exp(-((command[:, 2] - gyro[:, 2]) ** 2) / 0.25),
        "lin_vel_z": linvel[:, 2] ** 2,
        "ang_vel_xy": torch.sum(gyro[:, :2] ** 2, dim=1),
        "base_height": (backend.qpos[:, 2] - 0.3) ** 2,
        "action_rate": torch.sum(actions**2, dim=1),
        "similar_to_default": torch.sum(torch.abs(joint_pos - torch.tensor(_HOME)), dim=1),
    }
    for name, values in expected.items():
        torch.testing.assert_close(runtime.reward_components[name], values)

    contact = backend.sensors[:, 9:13] > 0.1
    phase = torch.tensor([0.04, 0.54, 0.54, 0.04]).repeat(3, 1)
    expected_contact = (contact == (phase < 0.6)).float().mean(dim=1)
    torch.testing.assert_close(runtime.reward_components["contact"], expected_contact)

    heights = torch.stack(
        (
            backend.sensors[:, 13],
            backend.sensors[:, 16],
            backend.sensors[:, 19],
            backend.sensors[:, 22],
        ),
        dim=1,
    )
    expected_swing = (torch.exp(-((heights - 0.1) ** 2) / 0.01) * (phase >= 0.6)).mean(dim=1)
    torch.testing.assert_close(runtime.reward_components["swing_feet_z"], expected_swing)
    assert not torch.any(state.terminated)
    assert not torch.any(state.truncated)


def test_go2_mask_reset_updates_true_rows_and_preserves_false_rows() -> None:
    backend, runtime = _make_runtime()
    tilt = 1.2
    tilted_qpos = backend.qpos.clone()
    tilted_qpos[0, 3:7] = torch.tensor(
        [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0], dtype=torch.float32
    )
    backend.qpos.copy_(tilted_qpos)
    backend._refresh_sensors(torch.tensor([True, False, False]))  # noqa: SLF001
    actions = (torch.arange(3, dtype=torch.float32).repeat(12, 1).t() * 0.1).contiguous()
    state = runtime.step(actions)
    assert state.terminated.tolist() == [True, False, False]

    reset_mask = state.terminated | state.truncated
    qpos_before = backend.qpos.clone()
    qvel_before = backend.qvel.clone()
    sensors_before = backend.sensors.clone()
    negotiated_sensors = backend.sensor_negotiations

    reset_obs = runtime.reset(reset_mask)

    assert reset_mask.tolist() == [True, False, False]
    assert backend.reset_calls == 2
    assert backend.sensor_negotiations == negotiated_sensors
    torch.testing.assert_close(backend.qpos[0], runtime.reset_qpos[0])
    torch.testing.assert_close(backend.qvel[0], runtime.reset_qvel[0])
    torch.testing.assert_close(backend.qpos[1:], qpos_before[1:])
    torch.testing.assert_close(backend.qvel[1:], qvel_before[1:])
    torch.testing.assert_close(backend.sensors[1:], sensors_before[1:])
    assert torch.count_nonzero(reset_obs["policy"][0, 30:42]) == 0
    assert torch.count_nonzero(reset_obs["policy"][1:, 30:42]) == 24
    assert runtime._steps.tolist() == [0, 1, 1]  # noqa: SLF001


def test_go2_all_false_reset_is_noop_and_preserves_every_row() -> None:
    backend, runtime = _make_runtime()
    actions = torch.full((3, 12), 0.2, dtype=torch.float32)
    runtime.step(actions)
    qpos_before = backend.qpos.clone()
    qvel_before = backend.qvel.clone()
    sensors_before = backend.sensors.clone()
    steps_before = runtime._steps.clone()  # noqa: SLF001
    reset_calls = backend.reset_calls

    obs = runtime.reset(torch.zeros(3, dtype=torch.bool))

    assert backend.reset_calls == reset_calls
    torch.testing.assert_close(backend.qpos, qpos_before)
    torch.testing.assert_close(backend.qvel, qvel_before)
    torch.testing.assert_close(backend.sensors, sensors_before)
    torch.testing.assert_close(runtime._steps, steps_before)  # noqa: SLF001
    assert obs["policy"].shape == (3, 49)
    assert obs["critic"].shape == (3, 52)


def test_go2_done_formula_parity_covers_bad_orientation_and_timeout() -> None:
    backend, runtime = _make_runtime(Go2TensorParityConfig(max_episode_steps=1))
    tilt = 1.2
    for row in (0, 2):
        backend.qpos[row, 3:7] = torch.tensor(
            [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0], dtype=torch.float32
        )
    backend._refresh_sensors(torch.tensor([True, False, True]))  # noqa: SLF001
    state = runtime.step(torch.zeros((3, 12), dtype=torch.float32))
    assert state.terminated.tolist() == [True, False, True]
    assert state.truncated.tolist() == [True, True, True]
    expected_angle = torch.arccos(backend.sensors[:, 8].clamp(-1, 1))
    torch.testing.assert_close(state.terminated, expected_angle > 1.0471975511965976)
    torch.testing.assert_close(state.truncated, torch.ones(3, dtype=torch.bool))


def test_go2_runtime_does_not_add_backend_registrations() -> None:
    from unilab.base import registry

    registry.ensure_registries()
    assert registry.list_registered_envs()["Go2JoystickFlat"]["available_backends"] == [
        "mujoco",
        "motrix",
        "superdex",
        "drake",
    ]


def test_go2_tensor_runtime_rejects_non_bool_reset_input() -> None:
    _, runtime = _make_runtime()
    with pytest.raises(TypeError, match="one-dimensional bool"):
        runtime.reset(torch.tensor([0, 0]))


def test_go2_tensor_runtime_rejects_wrong_reset_width() -> None:
    _, runtime = _make_runtime()
    with pytest.raises(ValueError, match="reset mask shape"):
        runtime.reset(torch.ones(2, dtype=torch.bool))


def test_go2_tensor_runtime_fails_closed_without_mask_capability() -> None:
    class NoMaskBackend(FakeGo2DeviceBackend):
        def get_tensor_capabilities(self):
            capabilities = super().get_tensor_capabilities()
            return TensorLifecycleCapabilities(
                **{**vars(capabilities), "mask_selected_reset": False}
            )

    backend = NoMaskBackend(num_envs=1)
    qpos, qvel = _reset_state(1)
    with pytest.raises(RuntimeError, match="mask tensor reset"):
        Go2TensorParityRuntime(
            backend,  # pyright: ignore[reportArgumentType]
            command=torch.zeros((1, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
        )


def test_go2_tensor_runtime_rejects_wrong_action_shape() -> None:
    _, runtime = _make_runtime()
    with pytest.raises(ValueError, match="actions shape"):
        runtime.step(torch.zeros((2, 12), dtype=torch.float32))


def test_go2_tensor_runtime_rejects_nonfinite_actions() -> None:
    _, runtime = _make_runtime()
    actions = torch.zeros((3, 12), dtype=torch.float32)
    actions[1, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        runtime.step(actions)


def test_go2_tensor_runtime_fails_closed_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeGo2DeviceBackend(num_envs=1)
    qpos, qvel = _reset_state(1)
    monkeypatch.setattr(
        tensor_parity_module.torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA Go2 tensor parity requested but CUDA is unavailable",
    ):
        Go2TensorParityRuntime(
            backend,  # pyright: ignore[reportArgumentType]
            command=torch.zeros((1, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
            device="cuda",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="wrong-device rejection requires an active CUDA context",
)
def test_go2_tensor_runtime_rejects_wrong_device_input() -> None:
    backend = FakeGo2DeviceBackend(num_envs=1)
    qpos, qvel = _reset_state(1)
    with pytest.raises(RuntimeError, match="did not accept Torch device"):
        Go2TensorParityRuntime(
            backend,  # pyright: ignore[reportArgumentType]
            command=torch.zeros((1, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
            device="cuda",
        )


def test_go2_tensor_runtime_rejects_mismatched_backend_layout() -> None:
    backend = FakeGo2DeviceBackend(num_envs=2)
    backend.qpos = torch.zeros((2, 18), dtype=torch.float32)
    qpos, qvel = _reset_state(2)
    with pytest.raises(ValueError, match="qpos view shape"):
        Go2TensorParityRuntime(
            backend,  # pyright: ignore[reportArgumentType]
            command=torch.zeros((2, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
        )


def test_go2_tensor_runtime_fails_closed_on_capability_mismatch() -> None:
    class NoSteppingBackend(FakeGo2DeviceBackend):
        def get_tensor_capabilities(self):
            capabilities = super().get_tensor_capabilities()
            return TensorLifecycleCapabilities(**{**vars(capabilities), "stepping": False})

    backend = NoSteppingBackend(num_envs=1)
    qpos, qvel = _reset_state(1)
    with pytest.raises(RuntimeError, match="tensor stepping"):
        Go2TensorParityRuntime(
            backend,  # pyright: ignore[reportArgumentType]
            command=torch.zeros((1, 3)),
            reset_qpos=qpos,
            reset_qvel=qvel,
        )
