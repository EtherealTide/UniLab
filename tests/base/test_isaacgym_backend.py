"""Protocol-level tests for the IsaacGym subprocess backend.

These tests drive ``IsaacGymBackend`` through ``isaacgym_mock_worker.py`` — a
deterministic kinematic fake that speaks the real wire protocol over real
shared memory on the host interpreter — so they run without the IsaacGym
runtime (real-runtime coverage lives in the slow conformance tests).
"""

from __future__ import annotations

import io
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from unisim.backend.isaacgym.backend import IsaacGymBackend, IsaacGymWorkerError
from unisim.backend.isaacgym.dependencies import (
    ENV_HOME,
    ENV_PYTHON,
    IsaacGymDependencyError,
    build_worker_env,
    resolve_isaacgym_runtime,
)
from unisim.backend.isaacgym.sensors import scan_scene_metadata
from unisim.backend.subprocess_ipc import protocol
from unisim.dr.types import INTERVAL_TERM_BODY_FORCE, INTERVAL_TERM_PUSH, IntervalTermOp

import unilab.base.backend_factory as backend_factory
from unilab.base.backend_factory import create_backend
from unilab.base.scene import SceneCfg


@pytest.fixture(autouse=True)
def _bypass_cuda_only_platform_gate(monkeypatch: pytest.MonkeyPatch):
    """Exercise the adapter below the factory's CUDA-only platform gate.

    The fail-closed gate (issue #1811 platform matrix) has dedicated coverage
    in ``tests/base/test_cuda_backend_platform_preflight.py``.  These
    mock-worker suites intentionally run on CUDA-less hosts, so the gate is
    bypassed here to keep the adapter behavior under test reachable.
    """
    monkeypatch.setattr(
        backend_factory, "_validate_cuda_only_backend_platform", lambda *args, **kwargs: None
    )


_MOCK_WORKER = str(Path(__file__).resolve().parent / "isaacgym_mock_worker.py")

SIM_DT = 0.005
NUM_ENVS = 2

_ROBOT_XML = """
<mujoco>
  <worldbody>
    <geom name="floor_geom" type="plane" size="0 0 0.05"/>
    <body name="base">
      <freejoint/>
      <site name="imu_site"/>
      <site name="tilted_site" pos="0.1 0 0" quat="0.7071068 0 0.7071068 0"/>
      <site name="euler_site" euler="0.1 0 0"/>
      <geom name="base_geom" size="0.1"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" size="0.1"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="g1" size="0.1"/>
          <body name="link2">
            <joint name="j2" type="hinge" range="-1.5 1.5"/>
            <geom name="g2" size="0.1"/>
            <geom name="foot_geom" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="10" kv="0.5" forcerange="-100 100" ctrlrange="-1.2 1.2"/>
    <position name="j1" joint="j1" kp="20" kv="1.0" forcerange="-80 80"/>
    <position name="j2" joint="j2" kp="30"/>
  </actuator>
</mujoco>
"""

_SCENE_XML = """
<mujoco>
  <include file="robot.xml"/>
  <sensor>
    <gyro name="base_gyro" site="imu_site"/>
    <velocimeter name="base_local_linvel" site="imu_site"/>
    <framequat name="link0_quat" objtype="body" objname="link0"/>
    <framepos name="link1_pos" objtype="body" objname="link1"/>
    <framepos name="tilted_site_pos" objtype="site" objname="tilted_site"/>
    <framezaxis name="base_upvector" objtype="body" objname="base"/>
    <framezaxis name="imu_upvector" objtype="site" objname="imu_site"/>
    <framezaxis name="tilted_upvector" objtype="site" objname="tilted_site"/>
    <framezaxis name="euler_upvector" objtype="site" objname="euler_site"/>
    <framezaxis name="geom_upvector" objtype="geom" objname="base_geom"/>
    <contact name="foot_contact" geom1="floor_geom" geom2="foot_geom" data="found" num="1"/>
    <accelerometer name="base_acc" site="imu_site"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture()
def scene_file(tmp_path: Path) -> str:
    (tmp_path / "robot.xml").write_text(textwrap.dedent(_ROBOT_XML), encoding="utf-8")
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(_SCENE_XML), encoding="utf-8")
    return str(scene)


def _make_backend(scene_file: str, **kwargs: Any) -> IsaacGymBackend:
    kwargs.setdefault("worker_command", [sys.executable, _MOCK_WORKER])
    kwargs.setdefault("worker_timeout_s", 30.0)
    base_name = kwargs.pop("base_name", "base")
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file),
        NUM_ENVS,
        SIM_DT,
        base_name=base_name,
        **kwargs,
    )
    assert isinstance(backend, IsaacGymBackend)
    return backend


def _state_numpy(backend: IsaacGymBackend, field: str) -> np.ndarray:
    return backend.get_state_views((field,))[field].detach().cpu().numpy()


@pytest.fixture()
def backend(scene_file: str) -> Any:
    instance = _make_backend(scene_file)
    instance.materialize()
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Protocol unit tests
# ---------------------------------------------------------------------------


def test_protocol_message_roundtrip() -> None:
    stream = io.BytesIO()
    protocol.send_message(stream, protocol.CMD_STEP, {"nsteps": 4})
    stream.seek(0)
    message = protocol.recv_message(stream)
    assert message == {"cmd": "STEP", "payload": {"nsteps": 4}}

    stream.seek(0)
    header = stream.read(protocol.HEADER_SIZE)
    assert protocol.unpack_header(header) == len(
        protocol.pack_message(protocol.CMD_STEP, {"nsteps": 4})
    )
    assert protocol.decode_message(stream.read())["cmd"] == "STEP"


def test_protocol_recv_eof_raises() -> None:
    with pytest.raises(protocol.WorkerDisconnectedError):
        protocol.recv_message(io.BytesIO(b""))
    with pytest.raises(protocol.WorkerDisconnectedError):
        protocol.recv_message(io.BytesIO(b"\x10\x00"))


def test_protocol_slot_layout() -> None:
    shapes = protocol.slot_shapes(num_envs=4, num_dof=3, num_bodies=2)
    assert shapes["ctrl"] == (4, 3)
    assert shapes["root_state"] == (4, 13)
    assert shapes["dof_state"] == (4, 3, 2)
    assert shapes["body_state"] == (4, 2, 13)
    assert shapes["contact_force"] == (4, 2, 3)
    assert shapes["reset_env_ids"] == (4,)
    assert shapes["reset_qpos"] == (4, 10)
    assert shapes["reset_qvel"] == (4, 9)
    assert protocol.slot_dtype("reset_env_ids") == np.dtype(np.int32)
    assert protocol.slot_nbytes("ctrl", shapes["ctrl"]) == 4 * 3 * 4
    with pytest.raises(ValueError, match="unknown shm slot"):
        protocol.slot_dtype("nope")
    with pytest.raises(ValueError, match="num_envs>0"):
        protocol.slot_shapes(0, 3, 2)


def test_protocol_error_payload() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        payload = protocol.serialize_exception(exc)
    assert payload["type"] == "RuntimeError"
    assert payload["message"] == "boom"
    assert "RuntimeError: boom" in payload["traceback"]
    rendered = protocol.format_worker_error(payload)
    assert "RuntimeError" in rendered and "boom" in rendered


def test_protocol_quat_helpers() -> None:
    xyzw = np.array([[0.1, 0.2, 0.3, 0.9]])
    wxyz = protocol.xyzw_to_wxyz(xyzw)
    np.testing.assert_allclose(wxyz, [[0.9, 0.1, 0.2, 0.3]])
    np.testing.assert_allclose(protocol.wxyz_to_xyzw(wxyz), xyzw)

    # 90-degree rotation about z maps +x to +y (and inverse maps +y to +x).
    half = np.sqrt(0.5)
    quat = np.array([[half, 0.0, 0.0, half]])
    np.testing.assert_allclose(
        protocol.quat_rotate(quat, np.array([[1.0, 0.0, 0.0]])), [[0.0, 1.0, 0.0]], atol=1e-7
    )
    np.testing.assert_allclose(
        protocol.quat_rotate_inverse(quat, np.array([[0.0, 1.0, 0.0]])),
        [[1.0, 0.0, 0.0]],
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------------


def _make_runtime_tree(home: Path) -> None:
    python = home / "miniconda3" / "envs" / "hsgym" / "bin" / "python3.8"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "miniconda3" / "envs" / "hsgym" / "lib").mkdir(parents=True)
    (home / "isaacgym" / "python").mkdir(parents=True)


def test_dependencies_resolve_default_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    _make_runtime_tree(tmp_path)
    runtime = resolve_isaacgym_runtime()
    assert runtime.python == tmp_path / "miniconda3" / "envs" / "hsgym" / "bin" / "python3.8"
    assert runtime.isaacgym_python == tmp_path / "isaacgym" / "python"
    env = build_worker_env(runtime)
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(
        tmp_path / "miniconda3" / "envs" / "hsgym" / "lib"
    )


def test_dependencies_python_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    _make_runtime_tree(tmp_path)
    custom = tmp_path / "custom_python"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PYTHON, str(custom))
    assert resolve_isaacgym_runtime().python == custom
    assert resolve_isaacgym_runtime(python_override=str(custom)).python == custom


def test_dependencies_missing_runtime_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "empty"))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    with pytest.raises(
        IsaacGymDependencyError, match="could not find the Python 3.8 worker interpreter"
    ):
        resolve_isaacgym_runtime()


# ---------------------------------------------------------------------------
# Mock-worker protocol tests
# ---------------------------------------------------------------------------


def test_materialize_binds_metadata_and_slots(backend: IsaacGymBackend) -> None:
    assert backend.num_envs == NUM_ENVS
    assert backend.num_actuators == 3
    assert backend.num_dof_vel == 3
    assert backend.get_actuator_names() == ("j0", "j1", "j2")
    assert backend.get_actuator_joint_names() == ("j0", "j1", "j2")
    np.testing.assert_allclose(
        backend.get_actuator_ctrl_range(),
        np.array([[-1.2, 1.2], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    kp, kd = backend.get_actuator_gains()
    np.testing.assert_allclose(kp, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(kd, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        backend.get_joint_range(), np.array([[-1.5, 1.5]] * 3, dtype=np.float32)
    )
    np.testing.assert_allclose(backend.get_gravity(), [0.0, 0.0, -9.81])
    np.testing.assert_array_equal(
        backend.get_body_ids(["base", "link1"]), np.array([0, 2], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="not found"):
        backend.get_body_ids(["missing_body"])

    default_qpos = backend.get_default_qpos()
    assert default_qpos.shape == (10,)
    # The sole scene keyframe ("home") is selected as the initial/default state.
    np.testing.assert_allclose(default_qpos, [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(backend.get_init_qvel(), np.zeros(9))
    np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(
        backend.get_keyframe_qpos("home"), [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3]
    )
    with pytest.raises(ValueError, match="Keyframe 'missing'"):
        backend.get_keyframe_qpos("missing")

    layout = backend.get_root_state_layout("base")
    assert layout.qpos_indices == tuple(range(7))
    assert layout.qvel_indices == tuple(range(6))
    with pytest.raises(NotImplementedError, match="root-state layout"):
        backend.get_root_state_layout("link0")

    np.testing.assert_array_equal(
        backend.get_joint_state_qpos_indices(["j1"]), np.array([8], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        backend.get_joint_state_qvel_indices(["j1"]), np.array([7], dtype=np.int32)
    )

    assert backend.model.num_bodies == 4


def test_tensor_state_requires_worker_gpu_pipeline(scene_file: str) -> None:
    """The CPU protocol mock deliberately fails closed outside CUDA IPC."""
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(RuntimeError, match="requires the worker GPU pipeline"):
            backend.get_state_views(("qpos",))
    finally:
        backend.close()


def test_xml_metadata_available_before_materialize(scene_file: str) -> None:
    """Keyframe/default qpos are pure parent-side XML metadata.

    Env construction (``resolve_scene_default_qpos``) queries them before
    ``materialize()`` spawns the worker; values must be identical afterwards.
    """
    backend = _make_backend(scene_file)
    expected_qpos = np.array([0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(backend.get_keyframe_qpos("home"), expected_qpos)
    np.testing.assert_allclose(backend.get_default_qpos(), expected_qpos)
    np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3])
    # Pure-XML metadata is available pre-materialize (env constructors read
    # these before the worker handshake).
    assert backend.num_actuators == 3
    assert backend.num_dof_vel == 3
    np.testing.assert_allclose(backend.get_init_qvel(), np.zeros(9))
    np.testing.assert_array_equal(
        backend.get_body_ids(["base", "link1"]), np.array([0, 2], dtype=np.int32)
    )
    assert backend.get_actuator_names() == ("j0", "j1", "j2")
    layout = backend.get_root_state_layout("base")
    assert layout.qpos_indices == tuple(range(7))
    # Worker-handshake-dependent surfaces materialize lazily on first use:
    # none of the pure-XML queries above spawned the worker.
    assert backend._proc is None
    _ = backend.model
    assert backend._proc is not None

    backend.materialize()
    try:
        np.testing.assert_allclose(backend.get_keyframe_qpos("home"), expected_qpos)
        np.testing.assert_allclose(backend.get_default_qpos(), expected_qpos)
        np.testing.assert_allclose(backend.get_default_dof_pos(), [0.1, 0.2, 0.3])
    finally:
        backend.close()


def test_actuator_metadata_available_before_materialize(scene_file: str) -> None:
    """PD gains / ctrl ranges / joint ranges are pure XML metadata.

    IsaacGym's MJCF importer drops kv/frictionloss/joint ranges, so these
    surfaces are answered from the parent-side scene scan and must not spawn
    the worker.
    """
    backend = _make_backend(scene_file)
    kp, kd = backend.get_actuator_gains()
    np.testing.assert_allclose(kp, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(kd, [0.5, 1.0, 0.0])
    np.testing.assert_allclose(
        backend.get_actuator_ctrl_range(),
        np.array([[-1.2, 1.2], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        backend.get_joint_range(), np.array([[-1.5, 1.5]] * 3, dtype=np.float32)
    )
    assert backend._proc is None


def test_scan_resolves_joint_default_classes(tmp_path: Path) -> None:
    """Joint armature/frictionloss resolve through MJCF default classes."""
    robot = """
    <mujoco>
      <default>
        <default class="robot">
          <joint armature="0.01" frictionloss="0.3"/>
        </default>
      </default>
      <worldbody>
        <body name="base" childclass="robot">
          <freejoint/>
          <body name="link0">
            <joint name="j0" type="hinge"/>
            <body name="link1">
              <joint name="j1" type="hinge" armature="0.5"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(robot), encoding="utf-8")
    metadata = scan_scene_metadata(str(scene))
    assert metadata.joint_names == ("j0", "j1")
    np.testing.assert_allclose(metadata.joint_armature, [0.01, 0.5])
    np.testing.assert_allclose(metadata.joint_frictionloss, [0.3, 0.3])
    assert np.isinf(metadata.joint_ranges[0]).all()


def test_scan_fails_closed_on_non_position_actuator(tmp_path: Path) -> None:
    """<motor>/other actuator types would break the position-target ctrl contract."""
    robot = textwrap.dedent(_ROBOT_XML).replace(
        "</mujoco>",
        '<actuator><motor name="m0" joint="j0"/></actuator>\n</mujoco>',
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(robot, encoding="utf-8")
    with pytest.raises(NotImplementedError, match="position"):
        scan_scene_metadata(str(scene))


def test_scan_fails_closed_on_asymmetric_forcerange(tmp_path: Path) -> None:
    """PhysX dof effort limits are symmetric; asymmetric MJCF forceranges fail."""
    robot = textwrap.dedent(_ROBOT_XML).replace(
        'forcerange="-100 100"',
        'forcerange="-50 100"',
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(robot, encoding="utf-8")
    with pytest.raises(NotImplementedError, match="asymmetric forcerange"):
        scan_scene_metadata(str(scene))


def test_init_keyframe_selection_prefers_default_keyframe_name(tmp_path: Path) -> None:
    scene = textwrap.dedent(_SCENE_XML).replace(
        '<key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>',
        '<key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2 0.3"/>'
        '<key name="crouch" qpos="0 0 0.5 1 0 0 0 -0.1 -0.2 -0.3"/>',
    )
    (tmp_path / "robot.xml").write_text(textwrap.dedent(_ROBOT_XML), encoding="utf-8")
    (tmp_path / "scene.xml").write_text(scene, encoding="utf-8")
    scene_file = str(tmp_path / "scene.xml")

    # Ambiguous keyframes without default_keyframe_name keep the zero default.
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        np.testing.assert_allclose(backend.get_default_dof_pos(), np.zeros(3))
    finally:
        backend.close()

    # An explicit default_keyframe_name selects that keyframe.
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file, default_keyframe_name="crouch"),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
        worker_command=[sys.executable, _MOCK_WORKER],
        worker_timeout_s=30.0,
    )
    assert isinstance(backend, IsaacGymBackend)
    backend.materialize()
    try:
        expected_qpos = np.array([0, 0, 0.5, 1, 0, 0, 0, -0.1, -0.2, -0.3], dtype=np.float32)
        np.testing.assert_allclose(backend.get_default_qpos(), expected_qpos)
    finally:
        backend.close()

    # A missing default_keyframe_name fails fast at materialize.
    backend = create_backend(
        "isaacgym",
        SceneCfg(model_file=scene_file, default_keyframe_name="missing"),
        NUM_ENVS,
        SIM_DT,
        base_name="base",
        worker_command=[sys.executable, _MOCK_WORKER],
        worker_timeout_s=30.0,
    )
    with pytest.raises(ValueError, match="default_keyframe_name 'missing' not found"):
        backend.materialize()
    backend.close()


def test_init_keyframe_dof_name_mismatch_fails(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_DOF_NAMES", "j0,j1,renamed_joint")
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(IsaacGymWorkerError, match="renamed_joint.*mjcf_joint_names"):
            backend.materialize()
    finally:
        backend.close()


def test_tensor_step_requires_worker_gpu_pipeline(backend: IsaacGymBackend) -> None:
    with pytest.raises(RuntimeError, match="requires the worker GPU pipeline"):
        backend.step_tensor(torch.ones((NUM_ENVS, 3), dtype=torch.float32))


def test_tensor_reset_requires_worker_gpu_pipeline(backend: IsaacGymBackend) -> None:
    rows = torch.tensor([0], dtype=torch.int64)
    qpos = torch.zeros((1, 10), dtype=torch.float32)
    qvel = torch.zeros((1, 9), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="requires the worker GPU pipeline"):
        backend.set_state_tensor(rows, qpos, qvel)


def test_sensor_mapping_from_scene_scan(scene_file: str) -> None:
    metadata = scan_scene_metadata(scene_file)
    assert metadata.sensors["base_gyro"].kind == "gyro"
    assert metadata.sensors["base_local_linvel"].kind == "local_linvel"
    assert metadata.sensors["link0_quat"].kind == "framequat"
    assert metadata.sensors["link1_pos"].kind == "framepos"
    assert metadata.sensors["foot_contact"].kind == "contact_found"
    assert "base_acc" in metadata.unsupported_sensors


def test_framezaxis_metadata_mapping(scene_file: str) -> None:
    metadata = scan_scene_metadata(scene_file)
    assert metadata.sensors["base_upvector"].kind == "framezaxis"
    assert metadata.sensors["tilted_upvector"].local_quat != (1.0, 0.0, 0.0, 0.0)
    assert metadata.sensors["tilted_site_pos"].local_pos == (0.1, 0.0, 0.0)
    assert "euler_upvector" in metadata.unsupported_sensors
    assert "geom_upvector" in metadata.unsupported_sensors


def test_tracked_body_views_fail_closed(backend: IsaacGymBackend) -> None:
    with pytest.raises(NotImplementedError, match="does not support tracked-body views"):
        backend.get_tracked_body_views(("base", "link1"), device="cpu")


def test_dr_and_pre_step_control_fail_closed(backend: IsaacGymBackend) -> None:
    capabilities = backend.get_dr_capabilities()
    assert not capabilities.supported_reset_terms
    assert not capabilities.supports_interval_term(INTERVAL_TERM_PUSH)
    assert not capabilities.supports_interval_term(INTERVAL_TERM_BODY_FORCE)

    class _Plan:
        def is_empty(self) -> bool:
            return False

        def iter_ops(self) -> tuple:
            return (IntervalTermOp("custom_unsupported", np.zeros(3)),)

    with pytest.raises(NotImplementedError, match="does not support interval term"):
        backend.apply_interval_randomization(_Plan())

    class _Payload:
        def is_empty(self) -> bool:
            return False

        def requested_terms(self) -> set[str]:
            return {"base_mass"}

    with pytest.raises(NotImplementedError, match="CUDA IPC reset randomization"):
        backend.set_state_tensor(
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, 10), dtype=torch.float32),
            torch.zeros((1, 9), dtype=torch.float32),
            randomization=_Payload(),
        )

    with pytest.raises(NotImplementedError, match="pre-step"):
        backend.set_pre_step_control(lambda backend_, ctrl: ctrl)
    backend.set_pre_step_control(None)

    # Physics-state playback is covered by the UniSim adapter contract tests.


def test_close_reaps_worker_and_unlinks_shm(backend: IsaacGymBackend) -> None:
    proc = backend._proc
    assert proc is not None and proc.poll() is None
    shm_names = [handle.name for handle in backend._shm_handles.values()]
    backend.close()
    assert proc.poll() is not None
    from multiprocessing import shared_memory

    for name in shm_names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)
    backend.close()  # idempotent
    with pytest.raises(IsaacGymWorkerError, match="closed or not materialized"):
        backend.step_tensor(torch.zeros((NUM_ENVS, 3), dtype=torch.float32))


def test_worker_init_error_propagates(scene_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "fail_init")
    backend = _make_backend(scene_file)
    try:
        with pytest.raises(IsaacGymWorkerError, match="mock init failure"):
            backend.materialize()
    finally:
        backend.close()


def test_constructor_rejects_bad_arguments(scene_file: str) -> None:
    with pytest.raises(ValueError, match="num_envs"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 0, SIM_DT)
    with pytest.raises(ValueError, match="sim_dt"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 1, -1.0, worker_command=["x"])
    with pytest.raises(TypeError, match="worker_command"):
        _make_backend(scene_file, worker_command="not-a-list")
    with pytest.raises(NotImplementedError, match="fragments"):
        create_backend(
            "isaacgym",
            SceneCfg(model_file=scene_file, fragment_files=["extra.xml"]),
            1,
            SIM_DT,
        )
    with pytest.raises(TypeError, match="does not accept backend options"):
        create_backend("isaacgym", SceneCfg(model_file=scene_file), 1, SIM_DT, bogus_option=1)


def test_g1_scene_framezaxis_sensors_resolve_from_real_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The g1_walk_flat scene declares the upvector sensors the policy needs.

    All four are ``objtype="site"`` frames in ``g1.xml``; the mock worker
    reports the owning bodies so the cold-path scan must resolve them to
    exact site-frame z axes.
    """
    from unisim.backend.isaacgym.sensors import scan_scene_metadata

    from unilab.assets import ASSETS_ROOT_PATH

    scene = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
    scene_metadata = scan_scene_metadata(scene)
    for name in (
        "pelvis_upvector",
        "torso_upvector",
        "left_foot_upvector",
        "right_foot_upvector",
    ):
        assert scene_metadata.sensors[name].kind == "framezaxis"


def test_sensor_binding_before_materialize_lazy_spawns_worker(scene_file: str) -> None:
    """Sensor binding auto-materializes while the constructor stays light."""
    backend = _make_backend(scene_file)
    try:
        assert backend._proc is None
        with pytest.raises(RuntimeError, match="requires the worker GPU pipeline"):
            backend.get_sensor_view("track_pos_w_base")
        assert backend._proc is not None
        # Explicit materialize() afterwards is a no-op.
        backend.materialize()
        with pytest.raises(RuntimeError, match="requires the worker GPU pipeline"):
            backend.get_sensor_view("track_pos_w_base")
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Native rendering / playback
# ---------------------------------------------------------------------------


def test_play_capabilities_advertise_native_rendering(backend: IsaacGymBackend) -> None:
    caps = backend.get_play_capabilities()
    assert caps.supports_native_interactive_renderer
    assert caps.supports_native_video_capture
    assert caps.supports_physics_state_playback


def test_normalize_camera_kwargs_maps_mujoco_convention() -> None:
    from unisim.backend.isaacgym.backend import _normalize_camera_kwargs

    out = _normalize_camera_kwargs(
        {"cam_distance": 3.0, "cam_elevation": -20.0, "cam_azimuth": 90.0}
    )
    assert out == {"distance": 3.0, "elevation_deg": 20.0, "azimuth_deg": 90.0}
    assert _normalize_camera_kwargs(None) == {
        "distance": 2.0,
        "elevation_deg": 20.0,
        "azimuth_deg": 90.0,
    }


def test_resolve_play_render_plan_modes(
    scene_file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = _make_backend(scene_file)
    try:
        plan = backend.resolve_play_render_plan(
            play_render_mode="none", play_steps=10, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "none" and plan.num_steps is None

        plan = backend.resolve_play_render_plan(
            play_render_mode="interactive", play_steps=None, output_video=None
        )
        assert plan.mode == "interactive" and not plan.headless and not plan.record_video

        plan = backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=5, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "record" and plan.headless and plan.record_video
        assert plan.num_steps == 5

        with pytest.raises(ValueError, match="play_steps"):
            backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=None, output_video=tmp_path / "x.mp4"
            )
        with pytest.raises(ValueError, match="output video path"):
            backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=5, output_video=None
            )

        monkeypatch.setenv("DISPLAY", ":0")
        plan = backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=5, output_video=None
        )
        assert plan.mode == "interactive"

        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        plan = backend.resolve_play_render_plan(
            play_render_mode="auto", play_steps=5, output_video=tmp_path / "x.mp4"
        )
        assert plan.mode == "record" and plan.headless
    finally:
        backend.close()


def test_interactive_renderer_roundtrip(backend: IsaacGymBackend) -> None:
    backend.init_renderer(headless=False)
    backend.render()
    # Re-initializing with the same config is a no-op; a different one fails.
    backend.init_renderer(headless=False)
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.init_renderer(headless=True, capture=True)
    # Capturing on an interactive-only renderer must not silently succeed.
    with pytest.raises(RuntimeError, match="already initialized"):
        backend.capture_video_frame()


def test_viewer_close_raises_render_closed(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unisim.backend.base import RenderClosedError

    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        backend.init_renderer(headless=False)
        with pytest.raises(RenderClosedError, match="closed"):
            backend.render()
    finally:
        backend.close()


def test_viewer_creation_failure_is_actionable(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "viewer_fails")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(IsaacGymWorkerError, match="create_viewer failed"):
            backend.init_renderer(headless=False)
    finally:
        backend.close()


def test_renderer_requires_graphics_context(
    scene_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "no_graphics")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        with pytest.raises(NotImplementedError, match="requires a GPU sim"):
            backend.init_renderer(headless=True, capture=True)
    finally:
        backend.close()


def test_capture_video_frame_roundtrip(backend: IsaacGymBackend) -> None:
    backend.init_renderer(headless=True, capture=True, width=64, height=48)
    frame = backend.capture_video_frame()
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    # Mock frame is a deterministic gradient: red follows rows, green columns.
    assert frame[10, 0, 0] == 10
    assert frame[0, 20, 1] == 20
    assert (frame[:, :, 2] == 128).all()


def test_capture_without_init_uses_record_config(backend: IsaacGymBackend) -> None:
    # capture_video_frame lazily initializes the headless capture camera.
    frame = backend.capture_video_frame()
    assert frame.shape == (720, 1280, 3)


def test_run_playback_record_writes_video(backend: IsaacGymBackend, tmp_path: Path) -> None:
    from types import SimpleNamespace

    output = tmp_path / "play.mp4"
    result = backend.run_playback(
        env=SimpleNamespace(cfg=None),
        initialize=lambda: 0,
        step=lambda obs: obs + 1,
        num_steps=3,
        output_video=output,
        headless=True,
        record_video=True,
    )
    assert result == str(output)
    assert output.exists() and output.stat().st_size > 0


def test_run_playback_interactive_finite_steps(backend: IsaacGymBackend) -> None:
    from types import SimpleNamespace

    steps: list[int] = []
    result = backend.run_playback(
        env=SimpleNamespace(cfg=None),
        initialize=lambda: 0,
        step=lambda obs: steps.append(obs) or (obs + 1),
        num_steps=2,
        headless=False,
        record_video=False,
    )
    assert result is None
    assert steps == [0, 1]


def test_run_playback_interactive_window_close_is_graceful(
    scene_file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("UNILAB_ISAACGYM_MOCK_BEHAVIOR", "close_on_render")
    backend = _make_backend(scene_file)
    backend.materialize()
    try:
        result = backend.run_playback(
            env=SimpleNamespace(cfg=None),
            initialize=lambda: 0,
            step=lambda obs: obs + 1,
            num_steps=None,
            headless=False,
            record_video=False,
        )
        assert result is None
    finally:
        backend.close()
