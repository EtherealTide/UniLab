"""Startup reset staging remains all-row and transactional."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unisim.backend.base import (
    PublicStateWidths,
    SelectedResetPublication,
    TensorDataPlane,
    TensorExecution,
    TensorLifecycleCapabilities,
    TensorProcessTopology,
)
from unisim.backend.packed_reset import PackedResetTransaction
from unisim.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unisim.scene_layout import CompiledSceneLayout, EntityLayout, JointLayout

from unilab.base.reset_state import ResetStateTransaction


def fixture(num_envs: int = 5):
    entity = EntityLayout(
        "object",
        "articulation",
        "floating",
        "base",
        ("base", "a", "b"),
        (0, 1, 2),
        (None, "base", "a"),
        (
            JointLayout("a", "hinge", (7,), (6,), "a"),
            JointLayout("b", "hinge", (8,), (7,), "b"),
        ),
        (),
        (),
        (),
        tuple(range(7)),
        tuple(range(6)),
    )
    layout = CompiledSceneLayout((entity,), 9, 8, 0, 3)
    reads: list[str] = []
    commits: list[object] = []
    state = {
        "root_pose": np.tile([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], (num_envs, 1)),
        "root_velocity": np.zeros((num_envs, 6)),
        "joint_positions": np.tile([0.2, 0.4], (num_envs, 1)),
        "joint_velocities": np.tile([0.6, 0.8], (num_envs, 1)),
    }

    def read(name: str):
        reads.append(name)
        return {key: values.copy() for key, values in state.items()}

    backend = SimpleNamespace(
        backend_type="fake",
        num_envs=num_envs,
        get_entity_state=read,
        reset_entities=commits.append,
        get_entity_default_state=lambda name, ids: {
            key: values[ids].copy() for key, values in state.items()
        },
        get_default_qpos=lambda: np.zeros(9),
        get_init_qvel=lambda: np.zeros(8),
        get_public_state_widths=lambda: PublicStateWidths(9, 8),
        get_tensor_capabilities=lambda: TensorLifecycleCapabilities(
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
            packed_reset=True,
            packed_reset_fields=frozenset((RESET_TERM_KP, RESET_TERM_KD)),
        ),
    )
    backend.bind_packed_reset = lambda *, device=None: PackedResetTransaction(
        backend, device=device
    )
    transaction = ResetStateTransaction(backend, scene_layout=layout)
    transaction.declare_packed_reset_device(torch.device("cpu"))
    return transaction, reads, commits


def test_startup_rejects_a_subset_of_rows() -> None:
    transaction, _, commits = fixture()
    with (
        pytest.raises(ValueError, match="every environment row"),
        transaction.scoped_startup(torch.tensor([1], dtype=torch.int64)),
    ):
        pass
    assert commits == []


def test_startup_write_commits_once_for_all_rows() -> None:
    transaction, reads, commits = fixture()
    all_rows = torch.arange(5, dtype=torch.int64)
    with transaction.scoped_startup(all_rows):
        transaction.write_entity_state(
            "object",
            all_rows,
            term_name="pose",
            root_pose=np.tile([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], (5, 1)),
        )
    assert reads == []
    assert len(commits) == 1


def test_startup_late_failure_commits_nothing() -> None:
    transaction, reads, commits = fixture()
    all_rows = torch.arange(5, dtype=torch.int64)
    with pytest.raises(ValueError), transaction.scoped_startup(all_rows):
        transaction.write_entity_state(
            "object",
            all_rows,
            term_name="pose",
            joint_names=("missing",),
            joint_positions=np.zeros((5, 1)),
        )
    assert reads == []
    assert commits == []
