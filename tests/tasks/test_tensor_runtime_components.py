from __future__ import annotations

import pytest
import torch
from unisim.backend.base import (
    HostBridgeTransferPlan,
    SelectedResetPublication,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)

from unilab.managers._noise.noise_cfg import UniformNoiseCfg
from unilab.tasks.motion_tracking.common.tensor_runtime import (
    EpisodeMetrics,
    ObservationNoise,
    ResetMasks,
    semantic_fingerprint,
)
from unilab.tasks.motion_tracking.common.tensor_state_store import DeviceStateStore


def test_semantic_fingerprint_rejects_semantic_mutation() -> None:
    base = {"std": 0.2, "enabled": True}
    mutated = dict(base)
    mutated["std"] = 0.25

    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.owner.v1", mutated
    )
    assert semantic_fingerprint("unilab.test.owner.v1", base) != semantic_fingerprint(
        "unilab.test.other.v1", base
    )


def test_semantic_fingerprint_rejects_non_finite_scalar() -> None:
    with pytest.raises(ValueError, match="non-finite scalar"):
        semantic_fingerprint("unilab.test.owner.v1", {"std": float("nan")})


def test_tensor_observation_noise_uses_declared_uniform_bounds() -> None:
    noise = ObservationNoise.from_uniform_terms(
        (
            UniformNoiseCfg(n_min=-0.1, n_max=0.2),
            UniformNoiseCfg(n_min=-2.0, n_max=-1.0),
        ),
        (2, 3),
        device=torch.device("cpu"),
    )
    observations = torch.zeros((4, 6))
    generator = torch.Generator().manual_seed(17)

    corrupted = noise.apply(observations, cursor=1, generator=generator)

    assert corrupted is observations
    assert bool((corrupted[:, 1:3] >= -0.1).all()) and bool((corrupted[:, 1:3] <= 0.2).all())
    assert bool((corrupted[:, 3:6] >= -2.0).all()) and bool((corrupted[:, 3:6] <= -1.0).all())
    torch.testing.assert_close(corrupted[:, 0], torch.zeros(4))


def test_tensor_reset_plan_publishes_full_width_done_union() -> None:
    plan = ResetMasks(
        terminated=torch.tensor([False, True, False, True]),
        truncated=torch.tensor([True, False, False, False]),
    )

    assert torch.equal(plan.done, torch.tensor([True, True, False, True]))


def test_tensor_reset_plan_fails_closed_on_mismatched_masks() -> None:
    with pytest.raises(ValueError, match="matching one-dimensional"):
        ResetMasks(
            terminated=torch.tensor([False, True]),
            truncated=torch.tensor([True, False, False]),
        )
    with pytest.raises(TypeError, match="boolean"):
        ResetMasks(
            terminated=torch.tensor([0, 1]),
            truncated=torch.tensor([1, 0]),
        )


def test_tensor_episode_metrics_include_terminal_then_reset() -> None:
    metrics = EpisodeMetrics.create(3, torch.device("cpu"))

    metrics.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False]))
    done_mask = torch.tensor([False, True, False])
    finished = metrics.finished_values(done_mask)
    metrics.reset(done_mask)
    metrics.update(torch.tensor([4.0, 5.0, 6.0]), torch.tensor([False, False, True]))

    torch.testing.assert_close(
        finished,
        torch.tensor([[0.0, 0.0], [2.0, 1.0], [0.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 5.0, 9.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 1, 2]))
    metrics.reset(torch.tensor([False, True, True]))
    torch.testing.assert_close(metrics.rewards, torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(metrics.lengths, torch.tensor([2, 0, 0]))


class _HostBridgeBackend:
    backend_type = "fake-host"

    def __init__(self):
        self.steps = 0

    def get_tensor_capabilities(self):
        return TensorLifecycleCapabilities(
            execution=TensorExecution.HOST_BRIDGE,
            packed_host_bridge=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.HOST_BRIDGE,
            stream_event_ownership="caller stream; fake synchronizes at host boundary",
            torch_devices=("cpu", "cuda"),
        )

    def bind_host_bridge_io(self, spec):
        backend = self

        class _Plan(HostBridgeTransferPlan):
            def __init__(self, plan_spec):
                self._spec = plan_spec
                self.last_timing = {}

            @property
            def spec(self):
                return self._spec

            @property
            def transfer_stats(self):
                return {}

            def read_state_sensors(self):
                return self._views()

            def write_control(self, ctrl):
                return None

            def step(self, nsteps=1):
                backend.steps += 1
                return None

            def apply_reset(self, env_indices, qpos, qvel, randomization=None):
                raise NotImplementedError("host-bridge reset is not mask-capable")

            def read_selected_state_sensors(self):
                raise NotImplementedError("host-bridge selected reads are unsupported")

            def close(self):
                return None

            def _views(self):
                state = backend.get_state_views(("qpos", "qvel"), device="cpu")
                state["pelvis_local_linvel"] = backend.get_sensor_view(
                    "pelvis_local_linvel", device="cpu"
                )
                state["torso_gyro"] = backend.get_sensor_view("torso_gyro", device="cpu")
                state["track_pos_w_pelvis"] = backend.get_sensor_view(
                    "track_pos_w_pelvis", device="cpu"
                )
                state["track_quat_w_pelvis"] = backend.get_sensor_view(
                    "track_quat_w_pelvis", device="cpu"
                )
                state["track_linvel_w_pelvis"] = backend.get_sensor_view(
                    "track_linvel_w_pelvis", device="cpu"
                )
                state["track_angvel_w_pelvis"] = backend.get_sensor_view(
                    "track_angvel_w_pelvis", device="cpu"
                )
                return state

        return _Plan(spec)

    def get_state_views(self, names, device=None):
        assert names == ("qpos", "qvel")
        return {
            "qpos": torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]],
                device=device,
            ),
            "qvel": torch.zeros((1, 7), device=device),
        }

    def get_sensor_view(self, name, device=None):
        values = {"pelvis_local_linvel": (1.0, 2.0, 3.0), "torso_gyro": (-1.0, -2.0, -3.0)}
        if name == "track_pos_w_pelvis":
            return torch.tensor([[0.1, 0.2, 0.3]], device=device)
        if name == "track_quat_w_pelvis":
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        if name in {"track_linvel_w_pelvis", "track_angvel_w_pelvis"}:
            return torch.zeros((1, 3), device=device)
        return torch.tensor(values[name], device=device).unsqueeze(0)


class _DeviceMaskResetBackend:
    backend_type = "fake-device-reset"

    def __init__(self):
        self.reset_calls = 0
        self.step_calls: list[tuple[torch.Tensor, int]] = []
        self.qpos = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25]])
        self.qvel = torch.zeros((1, 7))

    def get_tensor_capabilities(self):
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
            stream_event_ownership="backend invalidates derived views until readiness step",
            torch_devices=("cpu",),
        )

    def set_state_tensor_mask(self, reset_mask, qpos, qvel, randomization=None):
        assert randomization is None
        assert reset_mask.shape == (self.qpos.shape[0],)
        assert reset_mask.dtype is torch.bool
        self.reset_calls += 1
        self.qpos[reset_mask] = qpos[reset_mask]
        self.qvel[reset_mask] = qvel[reset_mask]
        return {"timing": {"mask_reset_ms": 1.0}}

    def step_tensor(self, ctrl, nsteps=1):
        self.step_calls.append((ctrl.detach().clone(), int(nsteps)))
        return {"timing": {"readiness_ms": 2.0}}

    def get_state_views(self, names, device=None):
        assert names == ("qpos", "qvel")
        return {"qpos": self.qpos, "qvel": self.qvel}

    def get_sensor_view(self, name, device=None):
        num_envs = self.qpos.shape[0]
        if name == "pelvis_local_linvel":
            return torch.tensor([[1.0, 2.0, 3.0]], device=device).repeat(num_envs, 1)
        if name == "torso_gyro":
            return torch.tensor([[-1.0, -2.0, -3.0]], device=device).repeat(num_envs, 1)
        if name == "track_pos_w_pelvis":
            return torch.tensor([[0.1, 0.2, 0.3]], device=device).repeat(num_envs, 1)
        if name == "track_quat_w_pelvis":
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
        if name in {"track_linvel_w_pelvis", "track_angvel_w_pelvis"}:
            return torch.zeros((num_envs, 3), device=device)
        return torch.tensor([1.0, 2.0, 3.0], device=device).repeat(num_envs, 1)


class _AdvancingNamedSensorBackend(_DeviceMaskResetBackend):
    """A DEVICE_RESIDENT adapter that recomputes named sensors on each read."""

    backend_type = "fake-device-advancing-sensors"

    def __init__(self):
        super().__init__()
        self.named_read_count = 0

    def get_sensor_view(self, name, device=None):
        if name in {"track_pos_w_pelvis", "track_quat_w_pelvis"}:
            return super().get_sensor_view(name, device=device)
        self.named_read_count += 1
        value = float(self.named_read_count)
        return torch.full((1, 3), value, device=device)


def test_tensor_state_store_full_read_validates_layout_and_finite_state() -> None:
    store = DeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )

    store.read()
    store.validate_finite()

    torch.testing.assert_close(store.joint_pos, torch.tensor([[0.25]]))
    torch.testing.assert_close(store.linvel, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(store.robot_body_pos, torch.tensor([[[0.1, 0.2, 0.3]]]))


def _device_store(backend: _DeviceMaskResetBackend | None = None):
    backend = backend or _DeviceMaskResetBackend()
    return backend, DeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )


def test_tensor_state_store_mask_reset_updates_true_rows_only() -> None:
    backend = _DeviceMaskResetBackend()
    backend.qpos = backend.qpos.repeat(2, 1)
    backend.qvel = backend.qvel.repeat(2, 1)
    store = DeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=2,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )
    store.read()
    qpos_before, qvel_before = store.qviews()
    reset_qpos = torch.full_like(qpos_before, -2.0)
    reset_qvel = torch.full_like(qvel_before, -3.0)
    all_false = torch.zeros(2, dtype=torch.bool)

    assert store.apply_reset(all_false, reset_qpos, reset_qvel) is None
    assert backend.reset_calls == 0
    torch.testing.assert_close(store.qpos, qpos_before)
    torch.testing.assert_close(store.qvel, qvel_before)

    reset_mask = torch.tensor([False, True])
    reset_result = store.apply_reset(reset_mask, reset_qpos, reset_qvel)

    assert reset_result == {"timing": {"mask_reset_ms": 1.0}}
    assert backend.reset_calls == 1
    torch.testing.assert_close(store.qpos[0], qpos_before[0])
    torch.testing.assert_close(store.qvel[0], qvel_before[0])
    torch.testing.assert_close(store.qpos[1], reset_qpos[1])
    torch.testing.assert_close(store.qvel[1], reset_qvel[1])


def test_tensor_state_store_reset_rejects_non_full_width_non_bool_masks() -> None:
    _, store = _device_store()
    qpos = torch.zeros((1, 8))
    qvel = torch.zeros((1, 7))
    with pytest.raises(ValueError, match="full-width bool"):
        store.apply_reset(torch.zeros(1, dtype=torch.int64), qpos, qvel)
    with pytest.raises(ValueError, match="full-width bool"):
        store.apply_reset(torch.zeros(2, dtype=torch.bool), qpos, qvel)


def test_tensor_state_store_crosses_named_sensor_read_boundary_each_read() -> None:
    backend = _AdvancingNamedSensorBackend()
    store = DeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )

    store.read()
    first = store.linvel.clone()
    store.read()
    second = store.linvel.clone()

    torch.testing.assert_close(first, torch.ones((1, 3)))
    torch.testing.assert_close(second, torch.full((1, 3), 5.0))
    torch.testing.assert_close(store.gyro, torch.full((1, 3), 6.0))


def test_tensor_state_store_reset_fails_closed_for_host_bridge_execution() -> None:
    store = DeviceStateStore(
        backend=_HostBridgeBackend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )
    store.read()

    with pytest.raises(NotImplementedError, match="DEVICE_RESIDENT"):
        store.apply_reset(torch.ones(1, dtype=torch.bool), torch.zeros((1, 8)), torch.zeros((1, 7)))


def test_tensor_state_store_reset_fails_closed_without_mask_capability() -> None:
    class NoMaskBackend(_DeviceMaskResetBackend):
        def get_tensor_capabilities(self):
            capabilities = super().get_tensor_capabilities()
            return TensorLifecycleCapabilities(
                **{**vars(capabilities), "mask_selected_reset": False}
            )

    _, store = _device_store(NoMaskBackend())
    with pytest.raises(NotImplementedError, match="mask_selected_reset"):
        store.apply_reset(torch.ones(1, dtype=torch.bool), torch.zeros((1, 8)), torch.zeros((1, 7)))


def test_tensor_state_store_has_no_legacy_host_state_fallback() -> None:
    backend = _HostBridgeBackend()
    store = DeviceStateStore(
        backend=backend,  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )
    legacy_methods = (
        "get_body_pos_w",
        "get_body_quat_w",
        "get_body_lin_vel_w",
        "get_body_ang_vel_w",
    )

    for method in legacy_methods:
        assert not hasattr(backend, method)

    store.read()


def test_tensor_state_store_rejects_unsupported_backend_before_read() -> None:
    class UnsupportedBackend:
        backend_type = "fake-unsupported"

        def get_tensor_capabilities(self):
            return TensorLifecycleCapabilities(execution=TensorExecution.UNSUPPORTED)

    with pytest.raises(RuntimeError, match="DEVICE_RESIDENT or HOST_BRIDGE"):
        DeviceStateStore(
            backend=UnsupportedBackend(),  # pyright: ignore[reportArgumentType]
            device=torch.device("cpu"),
            num_envs=1,
            joint_qpos_ids=torch.tensor([0], dtype=torch.int64),
            joint_qvel_ids=torch.tensor([0], dtype=torch.int64),
            body_names=("pelvis",),
            body_ids=torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("exact_device", [False, True])
def test_tensor_state_store_accepts_cuda_family_and_current_exact_device(exact_device) -> None:
    class Backend(_HostBridgeBackend):
        def get_tensor_capabilities(self):
            torch_devices = ("cuda",)
            if exact_device:
                torch_devices = (f"cuda:{torch.cuda.current_device()}",)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.HOST_BRIDGE,
                packed_host_bridge=True,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.HOST_BRIDGE,
                stream_event_ownership="caller stream; fake synchronizes at host boundary",
                torch_devices=torch_devices,
            )

    current = torch.device("cuda", index=torch.cuda.current_device())
    store = DeviceStateStore(
        backend=Backend(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cuda"),
        num_envs=1,
        joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
        joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
        body_names=("pelvis",),
        body_ids=torch.tensor([0], dtype=torch.int64),
    )

    assert store.device == current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_state_store_rejects_wrong_exact_cuda_device_before_allocation() -> None:
    class Backend:
        backend_type = "fake-wrong-cuda"

        def get_tensor_capabilities(self):
            wrong_index = (torch.cuda.current_device() + 1) % max(2, torch.cuda.device_count() + 1)
            return TensorLifecycleCapabilities(
                execution=TensorExecution.DEVICE_RESIDENT,
                process_topology=TensorProcessTopology.IN_PROCESS,
                data_plane=TensorDataPlane.DIRECT,
                stream_event_ownership="backend completes fake lifecycle",
                torch_devices=(f"cuda:{wrong_index}",),
            )

    with pytest.raises(RuntimeError, match="did not accept Tensor device"):
        DeviceStateStore(
            backend=Backend(),  # pyright: ignore[reportArgumentType]
            device=torch.device("cuda"),
            num_envs=1,
            joint_qpos_ids=torch.tensor([7], dtype=torch.int64),
            joint_qvel_ids=torch.tensor([6], dtype=torch.int64),
            body_names=("pelvis",),
            body_ids=torch.tensor([0], dtype=torch.int64),
        )
