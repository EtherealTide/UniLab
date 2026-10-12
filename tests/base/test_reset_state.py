"""Focused tests for the base-owned Manager-Based reset transaction."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    BackendMocapPoseBinding,
    BackendRootStateLayout,
    PublicStateWidths,
    SelectedResetPublication,
    SimBackend,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.dr.types import (
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_KD,
    RESET_TERM_KP,
    DomainRandomizationCapabilities,
    ResetRandomizationPayload,
)

from unilab.base.reset_state import ResetStateTransaction


class _Backend:
    backend_type = "fake"
    num_actuators = 3

    def __init__(
        self,
        *,
        qpos: Any = None,
        qvel: Any = None,
        fail_set_state: bool = False,
    ) -> None:
        self.num_envs = 4
        self.qpos = np.array([1.0, 2.0, 3.0]) if qpos is None else qpos
        self.qvel = np.array([0.0, 0.0]) if qvel is None else qvel
        self.fail_set_state = fail_set_state
        self.default_qpos_calls = 0
        self.init_qvel_calls = 0
        self.set_state_calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.randomization_calls: list[ResetRandomizationPayload | None] = []
        self.default_kp = np.array([10.0, 20.0, 30.0])
        self.default_kd = np.array([1.0, 2.0, 3.0])

    def get_default_qpos(self):
        self.default_qpos_calls += 1
        return self.qpos

    def get_init_qvel(self):
        self.init_qvel_calls += 1
        return self.qvel

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_KP, RESET_TERM_KD, "gravity"))
        )

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        return self.default_kp.copy(), self.default_kd.copy()

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_KP:
            return self.default_kp.copy()
        if term == RESET_TERM_KD:
            return self.default_kd.copy()
        if term == "gravity":
            return np.array([-1.0, 0.0, 9.81])
        raise NotImplementedError(term)

    def set_state(
        self,
        env_ids: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        randomization=None,
    ) -> dict:
        if self.fail_set_state:
            raise NotImplementedError("reset upload disabled")
        self.set_state_calls.append((env_ids.copy(), qpos.copy(), qvel.copy()))
        self.randomization_calls.append(randomization)
        return {"timing": {"set_state_ms": 1.0}}


class _TensorResetBackend(_Backend):
    """CPU fake exposing the public selected tensor-reset boundary."""

    def __init__(self) -> None:
        super().__init__(
            qpos=np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2]),
            qvel=np.zeros(8),
        )
        self.default_body_mass = np.array([2.0, 4.0])
        self.tensor_reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.tensor_mask_reset_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.tensor_mask_full_refresh_calls: list[bool] = []

    def tensor_execution(self):
        return TensorExecution.DEVICE_RESIDENT

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                (RESET_TERM_KP, RESET_TERM_KD, RESET_TERM_BODY_MASS, "gravity")
            )
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_BODY_MASS:
            return self.default_body_mass.copy()
        return super().get_reset_term_default(term)

    def get_tensor_capabilities(self):
        kwargs = {}
        if "mask_selected_reset" in TensorLifecycleCapabilities.__dataclass_fields__:
            kwargs["mask_selected_reset"] = True
            kwargs["mask_selected_reset_full_refresh"] = True
        return TensorLifecycleCapabilities(
            execution=TensorExecution.DEVICE_RESIDENT,
            state_views=True,
            state_fields=frozenset({"qpos", "qvel"}),
            sensor_views=False,
            stepping=True,
            selected_reset=True,
            reset_randomization=True,
            process_topology=TensorProcessTopology.IN_PROCESS,
            data_plane=TensorDataPlane.DIRECT,
            stream_event_ownership="fake synchronous stream",
            torch_devices=("cpu",),
            device_reset_randomization=True,
            selected_reset_publication=SelectedResetPublication.AUTHORITATIVE_VIEWS,
            **kwargs,
        )

    def set_state_tensor(self, env_indices, qpos, qvel, randomization=None) -> dict:
        rows = torch.as_tensor(env_indices, dtype=torch.int64)
        self.tensor_reset_calls.append((rows.clone(), qpos.clone(), qvel.clone(), randomization))
        return {"ok": True}

    def get_public_state_widths(self):
        return PublicStateWidths(len(self.qpos), len(self.qvel))

    def set_state_tensor_mask(
        self, reset_mask, qpos, qvel, randomization=None, full_refresh=False
    ) -> dict:
        assert randomization is None
        self.tensor_mask_reset_calls.append((reset_mask.clone(), qpos.clone(), qvel.clone()))
        self.tensor_mask_full_refresh_calls.append(bool(full_refresh))
        return {"ok": True}


def _transaction(backend: _Backend) -> ResetStateTransaction:
    return ResetStateTransaction(cast(SimBackend, backend))


class _ManipulationBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.poses = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (self.num_envs, 1))
        self.default_calls = 0

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset(
                (
                    "geom_size",
                    "geom_solref",
                    "geom_solimp",
                    "dof_damping",
                    "dof_frictionloss",
                )
            )
        )

    def _defaults(self, width):
        self.default_calls += 1
        return np.ones((3, width)) if width else np.ones(2)

    def get_reset_term_default(self, term: str) -> np.ndarray:
        widths = {
            "geom_size": 3,
            "geom_solref": 2,
            "geom_solimp": 5,
            "dof_damping": 0,
            "dof_frictionloss": 0,
        }
        if term not in widths:
            raise NotImplementedError(term)
        return self._defaults(widths[term])

    def set_state(self, env_ids, qpos, qvel, randomization=None):
        self.events.append("state")
        self.poses[env_ids, 0] = 0.0
        return super().set_state(env_ids, qpos, qvel, randomization)

    def bind_mocap_pose(self, body_name):
        def write(ids, poses):
            self.events.append("mocap")
            self.poses[ids] = poses

        return BackendMocapPoseBinding(
            self.backend_type,
            body_name,
            self.num_envs,
            self.poses[0].copy(),
            lambda: self.poses,
            write,
        )


class _PerWorldBodyMassBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.default_calls = 0

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_BODY_MASS,))
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term != RESET_TERM_BODY_MASS:
            raise NotImplementedError(term)
        self.default_calls += 1
        return np.asarray(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ]
        )


class _StartupModelFieldBackend(_PerWorldBodyMassBackend):
    """Expose canonical mass and CoM defaults for startup baseline tests."""

    def get_dr_capabilities(self):
        return DomainRandomizationCapabilities(
            supported_reset_terms=frozenset((RESET_TERM_BODY_MASS, RESET_TERM_BODY_IPOS))
        )

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term == RESET_TERM_BODY_IPOS:
            return np.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]], dtype=np.float64)
        return super().get_reset_term_default(term)


class _InvalidDefaultBackend(_Backend):
    def __init__(self, field: str, value: Any) -> None:
        super().__init__()
        self.field = field
        self.value = value

    def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
        return DomainRandomizationCapabilities(supported_reset_terms=frozenset((self.field,)))

    def get_reset_term_default(self, term: str) -> np.ndarray:
        if term != self.field:
            raise NotImplementedError(term)
        return self.value


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("body_mass", np.ones((2, 3), dtype=np.float64), "canonical 1-D table"),
        (
            "body_mass",
            np.ones((3, 2), dtype=np.float64),
            "canonical 1-D table or a per-environment \\(4, \\*canonical\\)",
        ),
        ("body_mass", np.ones(2, dtype=np.int32), "must be floating"),
        ("gravity", np.ones(2, dtype=np.float64), "canonical 1-D table"),
    ],
)
def test_reset_term_default_shapes_fail_closed(field: str, value: Any, match: str) -> None:
    transaction = _transaction(_InvalidDefaultBackend(field, value))
    with pytest.raises((TypeError, ValueError), match=match):
        if field == "gravity":
            transaction.bind_gravity_write(term_name="bad_default")
        else:
            transaction.bind_body_mass_write(
                np.array([0], dtype=np.int32),
                term_name="bad_default",
            )


def test_mask_reset_combines_default_and_joint_writes() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, True, False])

    with transaction.scoped_device_event_tensor(reset_mask):
        transaction.reset_to_default(reset_mask, term_name="scene")
        transaction.write_joint_state_mask(
            reset_mask,
            np.array([7, 8], dtype=np.int32),
            np.array([6, 7], dtype=np.int32),
            torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]),
            term_name="joints",
        )

    assert backend.tensor_reset_calls == []
    assert len(backend.tensor_mask_reset_calls) == 1
    committed_mask, qpos, qvel = backend.tensor_mask_reset_calls[0]
    torch.testing.assert_close(committed_mask, reset_mask)
    np.testing.assert_allclose(qpos[[0, 3], 7:], np.tile(backend.qpos[7:], (2, 1)))
    np.testing.assert_allclose(qvel[[0, 3], 6:], np.tile(backend.qvel[6:], (2, 1)))
    np.testing.assert_allclose(qpos[1, 7:], [0.1, 0.2])
    np.testing.assert_allclose(qpos[2, 7:], [0.3, 0.4])
    np.testing.assert_allclose(qvel[1, 6:], [1.0, 2.0])
    np.testing.assert_allclose(qvel[2, 6:], [3.0, 4.0])


def test_mask_reset_without_writes_does_not_commit() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))

    with transaction.scoped_device_event_tensor(torch.zeros(4, dtype=torch.bool)):
        pass

    assert backend.tensor_mask_reset_calls == []
    assert not transaction.last_commit_had_writes


def test_mask_reset_preserves_full_width_carrier_and_full_refresh_flag() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([True, False, True, False])
    layout = BackendRootStateLayout(tuple(range(7)), tuple(range(6)))
    root_state = torch.zeros((4, 13), dtype=torch.float32)
    root_state[:, 3] = 1.0
    root_state[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    with transaction.scoped_device_event_tensor(reset_mask, full_refresh=True):
        transaction.write_root_state_tensor(reset_mask, layout, root_state, term_name="root")

    committed_mask, qpos, _qvel = backend.tensor_mask_reset_calls[0]
    torch.testing.assert_close(committed_mask, reset_mask)
    assert backend.tensor_mask_full_refresh_calls == [True]
    np.testing.assert_allclose(
        qpos[:, 0], [0.1, backend.qpos[0], 0.3, backend.qpos[0]], rtol=0, atol=1e-7
    )


def test_mask_reset_randomization_fails_closed() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    reset_mask = torch.tensor([False, True, False, False])

    with pytest.raises(NotImplementedError, match="unsupported by the fixed-width mask reset"):
        with transaction.scoped_device_event_tensor(reset_mask):
            transaction.write_randomization_tensor_mask(
                RESET_TERM_BODY_MASS,
                reset_mask,
                np.array([0], dtype=np.int32),
                torch.zeros((4, 1)),
                term_name="randomize_rigid_body_mass",
            )

    assert backend.tensor_mask_reset_calls == []


def test_owner_tensor_subset_keeps_full_width_contract() -> None:
    backend = _TensorResetBackend()
    transaction = _transaction(backend)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    active_mask = torch.tensor([False, True, True, False])
    owner_mask = torch.tensor([False, True, False, False])

    with pytest.raises(ValueError, match="outside the active mask"):
        with transaction.scoped_device_owner_tensor(active_mask):
            transaction.write_joint_state_mask(
                torch.tensor([True, False, False, False]),
                np.array([7], dtype=np.int32),
                np.array([6], dtype=np.int32),
                torch.zeros((4, 1)),
                torch.zeros((4, 1)),
                term_name="outside_owner",
            )

    with transaction.scoped_device_owner_tensor(active_mask):
        transaction.write_joint_state_mask(
            owner_mask,
            np.array([7], dtype=np.int32),
            np.array([6], dtype=np.int32),
            torch.tensor([[0.0], [0.7], [0.9], [0.0]]),
            torch.tensor([[0.0], [1.7], [1.9], [0.0]]),
            term_name="owner",
        )

    committed_mask, qpos, qvel = backend.tensor_mask_reset_calls[0]
    torch.testing.assert_close(committed_mask, active_mask)
    np.testing.assert_allclose(
        qpos[:, 7],
        [backend.qpos[7], 0.7, backend.qpos[7], backend.qpos[7]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        qvel[:, 6],
        [backend.qvel[6], 1.7, backend.qvel[6], backend.qvel[6]],
        rtol=0,
        atol=1e-7,
    )


def test_startup_model_field_commit_becomes_reset_baseline() -> None:
    backend = _StartupModelFieldBackend()
    transaction = _transaction(backend)
    all_rows = torch.arange(backend.num_envs, dtype=torch.int64)
    columns = np.arange(3, dtype=np.int32)
    _, mass_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")
    startup_mass = np.asarray(mass_defaults) * 2.0
    _, ipos_defaults = transaction.bind_body_ipos_write(columns, term_name="startup_com")
    startup_ipos = np.broadcast_to(np.asarray(ipos_defaults), (backend.num_envs, 3, 3)) + 0.1

    with transaction.scoped_startup(torch.arange(backend.num_envs)):
        transaction.write_body_mass(
            all_rows,
            columns,
            startup_mass,
            term_name="startup_mass",
        )
        transaction.record_startup_randomization(
            RESET_TERM_BODY_MASS,
            all_rows,
            columns,
            startup_mass,
            term_name="startup_mass",
        )
        transaction.write_body_ipos(all_rows, columns, startup_ipos, term_name="startup_com")
        transaction.record_startup_randomization(
            RESET_TERM_BODY_IPOS,
            all_rows,
            columns,
            startup_ipos,
            term_name="startup_com",
        )

    startup_payload = backend.randomization_calls[-1]
    assert startup_payload is not None
    assert startup_payload.body_mass is not None
    assert startup_payload.body_ipos is not None


def test_startup_model_field_baseline_promotion_is_deferred_until_commit() -> None:
    backend = _StartupModelFieldBackend()
    backend.fail_set_state = True
    transaction = _transaction(backend)
    rows = torch.arange(backend.num_envs, dtype=torch.int64)
    columns = np.array([1], dtype=np.int32)
    _, selected_defaults = transaction.bind_body_mass_write(columns, term_name="startup_mass")

    with pytest.raises(NotImplementedError, match="reset upload disabled"):
        with transaction.scoped_startup(torch.arange(backend.num_envs)):
            transaction.write_body_mass(
                rows, columns, selected_defaults.clone(), term_name="startup_mass"
            )
            transaction.record_startup_randomization(
                RESET_TERM_BODY_MASS,
                rows,
                columns,
                selected_defaults.clone(),
                term_name="startup_mass",
            )

    # On failure, promotion did not occur: the default remains the backend's
    # per-environment construction table, not the staged startup rows.
    np.testing.assert_array_equal(
        transaction._randomization_defaults[RESET_TERM_BODY_MASS],
        backend.get_reset_term_default(RESET_TERM_BODY_MASS),
    )


@pytest.mark.parametrize(
    ("ids", "error", "match"),
    [
        ([0], TypeError, "must be torch.Tensor"),
        (torch.tensor([[0]], dtype=torch.int64), TypeError, "1-D integer"),
        (torch.tensor([True]), TypeError, "1-D integer"),
        (torch.tensor([-1], dtype=torch.int64), IndexError, "out of range"),
        (torch.tensor([4], dtype=torch.int64), IndexError, "out of range"),
        (torch.tensor([1, 1], dtype=torch.int64), ValueError, "contain duplicates"),
    ],
)
def test_begin_rejects_invalid_environment_ids(ids, error, match: str) -> None:
    with pytest.raises(error, match=match):
        _transaction(_Backend()).begin(ids)


def _bind_device_mass(transaction: ResetStateTransaction) -> None:
    transaction.bind_body_mass_write(np.array([0, 1], dtype=np.int32), term_name="mass")
