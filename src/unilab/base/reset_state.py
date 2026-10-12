"""Base-owned reset-state transaction for Manager-Based event terms.

The transaction composes NumPy state writes in memory and hands the finished
batch to :meth:`SimBackend.set_state` exactly once.  It deliberately knows
nothing about task configuration, IPC, runners, or backend-private state.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch
from unisim.backend.base import (
    BackendMocapPoseBinding,
    BackendRootStateLayout,
    SimBackend,
    TensorExecution,
)
from unisim.dr.types import (
    RESET_TERM_BODY_INERTIA,
    RESET_TERM_BODY_IPOS,
    RESET_TERM_BODY_MASS,
    RESET_TERM_DOF_ARMATURE,
    RESET_TERM_DOF_DAMPING,
    RESET_TERM_DOF_FRICTIONLOSS,
    RESET_TERM_GEOM_FRICTION,
    RESET_TERM_GEOM_SIZE,
    RESET_TERM_GEOM_SOLIMP,
    RESET_TERM_GEOM_SOLREF,
    RESET_TERM_GRAVITY,
    RESET_TERM_KD,
    RESET_TERM_KP,
    ResetRandomizationPayload,
)
from unisim.entities import EntityStatePatch, SceneResetRequest
from unisim.scene_layout import CompiledSceneLayout

from unilab.base.rotation import np_quat_apply_inverse

_RANDOMIZATION_TERM_TAILS: dict[str, tuple[int, ...]] = {
    RESET_TERM_BODY_INERTIA: (3,),
    RESET_TERM_BODY_MASS: (),
    RESET_TERM_BODY_IPOS: (3,),
    RESET_TERM_DOF_ARMATURE: (),
    RESET_TERM_DOF_DAMPING: (),
    RESET_TERM_DOF_FRICTIONLOSS: (),
    RESET_TERM_GEOM_FRICTION: (3,),
    RESET_TERM_GEOM_SIZE: (3,),
    RESET_TERM_GEOM_SOLIMP: (5,),
    RESET_TERM_GEOM_SOLREF: (2,),
    RESET_TERM_GRAVITY: (),
    RESET_TERM_KD: (),
    RESET_TERM_KP: (),
}


def _randomization_term_tail(field: str) -> tuple[int, ...]:
    try:
        return _RANDOMIZATION_TERM_TAILS[field]
    except KeyError as exc:
        raise ValueError(f"unknown reset randomization term {field!r}") from exc


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    if result.flags.writeable:
        result = result.copy()
    result.setflags(write=False)
    return result


class ResetStateTransaction:
    """Reusable, fail-closed transaction for reset-mode state mutation."""

    def __init__(
        self,
        backend: SimBackend,
        *,
        default_qpos: np.ndarray | None = None,
        scene_layout: CompiledSceneLayout | None = None,
    ) -> None:
        self._backend = backend
        self._num_envs = backend.num_envs
        self._selected_default_qpos = default_qpos
        self.scene_layout = scene_layout
        if scene_layout is not None and any(
            joint.kind not in ("hinge", "slide")
            for entity in scene_layout.entities
            for joint in entity.joints
        ):
            raise NotImplementedError("mapped manager reset currently supports scalar joints")
        self._entity_values: dict[str, dict[str, np.ndarray]] = {}
        self._entity_fields: dict[str, set[str]] = {}
        self._entity_joints: dict[str, set[str]] = {}
        self._entity_joint_fields: dict[str, dict[str, set[str]]] = {}
        self._entity_layouts = (
            {}
            if scene_layout is None
            else {entity.name: entity for entity in scene_layout.entities}
        )
        self._entity_joint_columns = {
            name: {joint.name: index for index, joint in enumerate(entity.joints)}
            for name, entity in self._entity_layouts.items()
        }
        self._entity_rows: tuple[int, ...] | None = None
        self._entity_row_index: dict[int, int] = {}
        self._restore_entity_controls = False
        self._active = False
        self._tensor_active = False
        self._tensor_active_mask: torch.Tensor | None = None
        self._mask_full_refresh = False
        self._tensor_owner_subset = False
        self._active_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._dirty_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._default_qpos: np.ndarray | None = None
        self._default_qvel: np.ndarray | None = None
        self._qpos: np.ndarray | None = None
        self._qvel: np.ndarray | None = None
        self._tensor_qpos: torch.Tensor | None = None
        self._tensor_qvel: torch.Tensor | None = None
        self._tensor_qpos_default: torch.Tensor | None = None
        self._tensor_qvel_default: torch.Tensor | None = None
        self._default_kp: np.ndarray | None = None
        self._default_kd: np.ndarray | None = None
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._gain_dirty_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._randomization_defaults: dict[str, np.ndarray] = {}
        self._randomization_values: dict[str, np.ndarray] = {}
        self._randomization_dirty_masks: dict[str, np.ndarray] = {}
        self._startup_randomization_fields: set[str] = set()
        self._committed_randomization: dict[str, np.ndarray] = {}
        self._committed_randomization_masks: dict[str, np.ndarray] = {}
        self._committed_kp: np.ndarray | None = None
        self._committed_kd: np.ndarray | None = None
        self._committed_gain_mask = np.zeros(self._num_envs, dtype=np.bool_)
        self._requesting_terms: set[str] = set()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms: dict[str, float] = {}
        self._mocap_bindings: dict[str, BackendMocapPoseBinding] = {}
        self._mocap_values: dict[str, np.ndarray] = {}
        self._mocap_masks: dict[str, np.ndarray] = {}
        self._packed_reset_device: torch.device | None = None
        self._tensor_dr_columns: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self._tensor_dr_committed: dict[str, torch.Tensor] = {}
        self._tensor_root_columns: (
            tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        self._tensor_joint_columns: (
            tuple[tuple[int, ...], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None

    @property
    def active(self) -> bool:
        """Whether a reset lifecycle currently owns the transaction."""
        return self._active or self._tensor_active

    @property
    def tensor_active(self) -> bool:
        """Whether the active reset composes device tensor rows without host staging."""
        return self._tensor_active

    @property
    def last_commit_had_writes(self) -> bool:
        """Whether the most recent scoped commit submitted dirty rows to set_state."""
        return self._last_commit_had_writes

    @property
    def last_set_state_timing_ms(self) -> dict[str, float]:
        """Sub-timings from the most recent commit's set_state call.

        Always includes ``dr_reset_set_state_ms`` (outer wall-clock around the
        backend call); backend-reported ``set_state_*_ms`` sub-keys are merged
        in when the backend returns them. Empty when the last commit had no
        dirty rows.
        """
        return self._last_set_state_timing_ms

    @contextmanager
    def scoped_startup(self, env_ids: torch.Tensor) -> Iterator[ResetStateTransaction]:
        """Begin the all-row startup transaction and commit after all terms succeed."""
        ids = self._validate_ids(env_ids, capability="startup")
        expected_ids = np.arange(self._num_envs, dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            raise ValueError("startup reset-state transaction must include every environment row")
        self.begin(env_ids)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit()

    def declare_packed_reset_device(self, device: torch.device) -> None:
        """Declare the scene read-plan device paired with packed reset commits."""
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", index=torch.cuda.current_device())
        self._packed_reset_device = resolved

    @contextmanager
    def scoped_device_event_tensor(
        self, reset_mask: torch.Tensor, *, full_refresh: bool = False
    ) -> Iterator[ResetStateTransaction]:
        """Begin a fixed-width owner reset and commit it."""
        self.begin_tensor(reset_mask, full_refresh=full_refresh)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit_device_tensor()

    @contextmanager
    def scoped_device_owner_tensor(
        self, reset_mask: torch.Tensor
    ) -> Iterator[ResetStateTransaction]:
        """Begin a full-width step-owner tensor transaction."""
        self.begin_tensor(reset_mask, owner_subset=True)
        try:
            yield self
        except BaseException:
            self.abort()
            raise
        else:
            self.commit_device_tensor()

    def begin(self, env_ids: torch.Tensor) -> None:
        """Open a transaction for the concrete reset environment IDs."""
        if self._active:
            raise RuntimeError("ManagerBased reset-state transaction is already active")
        ids = self._validate_ids(env_ids, capability="begin")
        self._active_mask.fill(False)
        self._active_mask[ids] = True
        self._dirty_mask.fill(False)
        self._gain_dirty_mask.fill(False)
        for mask in self._randomization_dirty_masks.values():
            mask.fill(False)
        self._requesting_terms.clear()
        self._last_commit_had_writes = False
        self._last_set_state_timing_ms = {}
        self._entity_values.clear()
        self._entity_fields.clear()
        self._entity_joints.clear()
        self._entity_joint_fields.clear()
        self._entity_rows = None
        self._entity_row_index.clear()
        self._restore_entity_controls = False
        self._tensor_qpos = None
        self._tensor_qvel = None
        self._active = True

    def begin_tensor(
        self,
        reset_mask: torch.Tensor,
        *,
        owner_subset: bool = False,
        full_refresh: bool = False,
    ) -> None:
        """Open a fixed-width mask reset transaction."""
        if self._active or self._tensor_active:
            raise RuntimeError("ManagerBased reset-state transaction is already active")
        device = self._packed_reset_device
        if device is None:
            raise RuntimeError(
                "tensor reset transaction requires its selected device to be declared"
            )
        if not isinstance(reset_mask, torch.Tensor):
            raise TypeError("fixed-width tensor reset requires a Torch bool mask")
        if reset_mask.shape != (self._num_envs,) or reset_mask.dtype != torch.bool:
            raise ValueError(
                f"fixed-width tensor reset mask must have shape {(self._num_envs,)} and bool dtype"
            )
        if reset_mask.device != device:
            raise ValueError(
                "fixed-width tensor reset mask must live on the declared reset device "
                f"{device}; got {reset_mask.device}"
            )
        self._materialize_tensor_default_state("begin_tensor")
        self._reset_tensor_staging_mask(reset_mask)
        self._tensor_active_mask = reset_mask
        self._mask_full_refresh = bool(full_refresh)
        self._tensor_has_writes = False
        self._tensor_owner_subset = owner_subset
        self._requesting_terms.clear()
        self._tensor_active = True

    def write_entity_state(
        self,
        entity: str,
        env_ids: torch.Tensor,
        *,
        term_name: str,
        root_pose: np.ndarray | None = None,
        root_velocity: np.ndarray | None = None,
        joint_positions: np.ndarray | None = None,
        joint_velocities: np.ndarray | None = None,
        joint_names: tuple[str, ...] = (),
    ) -> None:
        """Stage one public physical-entity patch in the current manager transaction."""
        self._require_active()
        if self.scene_layout is None:
            raise NotImplementedError("physical entity patches require a compiled scene layout")
        ids = self._validate_ids(env_ids, capability=term_name)
        if np.any(~self._active_mask[ids]):
            raise ValueError("entity write is outside the active reset")
        if not len(ids):
            return
        patch = EntityStatePatch(
            entity, root_pose, root_velocity, joint_positions, joint_velocities, joint_names
        )
        request = SceneResetRequest(tuple(int(i) for i in ids), (patch,))
        self.scene_layout.validate_reset(request, num_envs=self._num_envs)
        rows = tuple(sorted(request.env_ids))
        if self._entity_rows is not None and self._entity_rows != rows:
            raise NotImplementedError(
                "one entity transaction requires the same selected env rows for every patch"
            )
        self._entity_rows = rows
        self._entity_row_index = {value: index for index, value in enumerate(rows)}
        incoming_rows = {value: index for index, value in enumerate(request.env_ids)}
        order = [incoming_rows[i] for i in rows]
        owner = self._entity_layouts[entity]
        if entity not in self._entity_values:
            self._entity_values[entity] = {}
            self._entity_fields[entity] = set()
            self._entity_joints[entity] = set()
            self._entity_joint_fields[entity] = {}
        values = self._entity_values[entity]
        selected_joints = joint_names or tuple(joint.name for joint in owner.joints)
        for field in ("root_pose", "root_velocity", "joint_positions", "joint_velocities"):
            incoming = getattr(patch, field)
            if incoming is None:
                continue
            if field.startswith("joint"):
                columns = [self._entity_joint_columns[entity][name] for name in selected_joints]
                if field not in values:
                    values[field] = np.empty((len(rows), len(owner.joints)), dtype=np.float64)
                values[field][:, columns] = incoming[order]
                self._entity_joints[entity].update(selected_joints)
                self._entity_joint_fields[entity].setdefault(field, set()).update(selected_joints)
            else:
                values[field] = incoming[order]
            self._entity_fields[entity].add(field)
        self._requesting_terms.add(term_name)

    def read_entity_root_pose(self, entity: str, env_ids: torch.Tensor) -> np.ndarray:
        """Read staged root pose or a detached current public snapshot."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="read_entity_root_pose")
        if np.any(~self._active_mask[ids]):
            raise ValueError("entity state read is outside the active reset")
        if "root_pose" in self._entity_values.get(entity, {}):
            rows = [self._entity_row_index[int(i)] for i in ids]
            return self._entity_values[entity]["root_pose"][rows].copy()
        return self._backend.get_entity_state(entity)["root_pose"][ids].copy()

    def _commit_entities(self) -> None:
        assert self.scene_layout is not None
        if self._entity_rows is None:
            return
        if np.any(self._dirty_mask) or any(np.any(mask) for mask in self._mocap_masks.values()):
            raise NotImplementedError("cannot mix entity patches with legacy state/DR writes")
        patches = []
        for name, values in self._entity_values.items():
            entity = self._entity_layouts[name]
            joint_names = tuple(
                j.name for j in entity.joints if j.name in self._entity_joints[name]
            )
            columns = [self._entity_joint_columns[name][n] for n in joint_names]
            current = None
            for field, written in self._entity_joint_fields[name].items():
                missing = self._entity_joints[name] - written
                if missing:
                    if current is None:
                        current = self._backend.get_entity_state(name)
                    absent = [self._entity_joint_columns[name][joint] for joint in missing]
                    values[field][:, absent] = current[field][np.ix_(self._entity_rows, absent)]
            fields = {
                field: values[field][:, columns] if field.startswith("joint") else values[field]
                for field in self._entity_fields[name]
            }
            patches.append(EntityStatePatch(name, joint_names=joint_names, **fields))
        request = SceneResetRequest(
            self._entity_rows,
            tuple(patches),
            restore_default_controls=self._restore_entity_controls,
        )
        self.scene_layout.validate_reset(request, num_envs=self._num_envs)
        started = time.perf_counter()
        self._backend.reset_entities(request)
        self._last_commit_had_writes = True
        self._last_set_state_timing_ms = {
            "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000
        }

    def bind_geom_size_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_size defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SIZE,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SIZE),
            capability="geom_size IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SIZE,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_size(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_size values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SIZE,
            env_ids,
            column_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def bind_geom_solref_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_solref defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SOLREF,
            expected_tail=(2,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SOLREF),
            capability="geom_solref IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SOLREF,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_solref(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_solref values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SOLREF,
            env_ids,
            column_ids,
            values,
            value_tail=(2,),
            term_name=term_name,
        )

    def bind_geom_solimp_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable geom_solimp defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_SOLIMP,
            expected_tail=(5,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_SOLIMP),
            capability="geom_solimp IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_SOLIMP,
        )
        return self._readonly_binding(columns, selected)

    def write_geom_solimp(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected geom_solimp values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_SOLIMP,
            env_ids,
            column_ids,
            values,
            value_tail=(5,),
            term_name=term_name,
        )

    def bind_dof_damping_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable dof_damping defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_DAMPING,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_DOF_DAMPING),
            capability="dof_damping IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_DAMPING,
        )
        return self._readonly_binding(columns, selected)

    def write_dof_damping(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected dof_damping values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_DOF_DAMPING,
            env_ids,
            column_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def bind_dof_frictionloss_write(
        self,
        column_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bind immutable dof_frictionloss defaults through the declared backend capability."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_FRICTIONLOSS,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(
                default,
                field=RESET_TERM_DOF_FRICTIONLOSS,
            ),
            capability="dof_frictionloss IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_FRICTIONLOSS,
        )
        return self._readonly_binding(columns, selected)

    def write_dof_frictionloss(
        self,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected dof_frictionloss values in the active reset."""
        self._write_selected_randomization(
            RESET_TERM_DOF_FRICTIONLOSS,
            env_ids,
            column_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def bind_body_mass_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body-mass columns and immutable backend defaults on the cold path."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_MASS,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_BODY_MASS),
            capability="body mass IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_BODY_MASS,
        )
        return columns, self._torch_default_table(selected)

    def bind_body_ipos_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body inertial-position columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_BODY_IPOS,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_BODY_IPOS),
            capability="body ipos IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_BODY_IPOS,
        )
        return columns, self._torch_default_table(selected)

    def bind_body_inertia_write(
        self,
        body_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind body-inertia columns and authoritative backend defaults.

        UniSim returns either a canonical ``(nbody, 3)`` table or a per-world
        ``(num_envs, nbody, 3)`` table. No caller-side model compilation or
        body-order cross-check is needed.
        """
        inertia = self._materialize_randomization_default(
            RESET_TERM_BODY_INERTIA,
            expected_tail=(3,),
            term_name=term_name,
        )
        if np.any(inertia < 0.0):
            raise ValueError(
                f"EventManager term '{term_name}' default body_inertia contains negative values"
            )
        columns = self._validate_columns(
            body_ids,
            width=self._randomization_default_width(default=inertia, field=RESET_TERM_BODY_INERTIA),
            capability="body inertia IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            inertia,
            columns,
            field=RESET_TERM_BODY_INERTIA,
        )
        return columns, self._torch_default_table(selected)

    def bind_gravity_write(self, *, term_name: str) -> np.ndarray:
        """Bind the immutable backend gravity vector on the cold path."""
        return self._materialize_randomization_default(
            RESET_TERM_GRAVITY,
            expected_tail=(),
            term_name=term_name,
        )

    def bind_gravity_write_torch(self, *, term_name: str, device: torch.device) -> torch.Tensor:
        """Bind the immutable backend gravity vector as a Torch table."""
        default = self.bind_gravity_write(term_name=term_name)
        return torch.from_numpy(np.array(default, copy=True)).to(device=device, dtype=torch.float64)

    def bind_dof_armature_write(
        self,
        dof_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind DOF-armature columns and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_DOF_ARMATURE,
            expected_tail=(),
            term_name=term_name,
        )
        columns = self._validate_columns(
            dof_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_DOF_ARMATURE),
            capability="DOF armature IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_DOF_ARMATURE,
        )
        return columns, self._torch_default_table(selected)

    def bind_geom_friction_write(
        self,
        geom_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Bind geom-friction rows and immutable backend defaults."""
        default = self._materialize_randomization_default(
            RESET_TERM_GEOM_FRICTION,
            expected_tail=(3,),
            term_name=term_name,
        )
        columns = self._validate_columns(
            geom_ids,
            width=self._randomization_default_width(default, field=RESET_TERM_GEOM_FRICTION),
            capability="geom friction IDs",
            term_name=term_name,
        )
        selected = self._select_randomization_default_columns(
            default,
            columns,
            field=RESET_TERM_GEOM_FRICTION,
        )
        return columns, self._torch_default_table(selected)

    def _torch_default_table(self, values: np.ndarray) -> torch.Tensor:
        """Materialize an immutable backend default as a Torch table."""
        return torch.from_numpy(np.array(values, dtype=np.float64, copy=True)).to(
            dtype=torch.float64
        )

    def write_body_mass(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body masses in the exactly-once reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_MASS,
            env_ids,
            body_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_body_ipos(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body inertial positions in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_IPOS,
            env_ids,
            body_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def write_body_inertia(
        self,
        env_ids: torch.Tensor,
        body_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected body principal-inertia diagonals in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_BODY_INERTIA,
            env_ids,
            body_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def write_gravity(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage per-environment gravity vectors in the reset payload."""
        ids = self._prepare_state_write(
            env_ids,
            capability="gravity",
            term_name=term_name,
        )
        default = self._require_randomization_default(RESET_TERM_GRAVITY, term_name)
        gravity = self._validate_values(
            values,
            shape=(ids.size, 3),
            capability="gravity",
            term_name=term_name,
        )
        buffer = self._randomization_values[RESET_TERM_GRAVITY]
        mask = self._randomization_dirty_masks[RESET_TERM_GRAVITY]
        uninitialized = ids[~mask[ids]]
        if uninitialized.size:
            buffer[uninitialized] = self._randomization_default_rows(
                default,
                uninitialized,
                field=RESET_TERM_GRAVITY,
            )
        buffer[ids] = gravity
        mask[ids] = True
        self._dirty_mask[ids] = True

    def write_dof_armature(
        self,
        env_ids: torch.Tensor,
        dof_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected DOF armatures in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_DOF_ARMATURE,
            env_ids,
            dof_ids,
            values,
            value_tail=(),
            term_name=term_name,
        )

    def write_geom_friction(
        self,
        env_ids: torch.Tensor,
        geom_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected three-axis geom friction in the reset payload."""
        self._write_selected_randomization(
            RESET_TERM_GEOM_FRICTION,
            env_ids,
            geom_ids,
            values,
            value_tail=(3,),
            term_name=term_name,
        )

    def bind_actuator_gain_write(
        self,
        actuator_ids: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Resolve gain mutation capability and immutable defaults on the cold path."""
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        assert self._default_kp is not None
        assert self._default_kd is not None
        selected_kp = self._torch_default_table(
            self._select_randomization_default_columns(
                self._default_kp,
                columns,
                field=RESET_TERM_KP,
            )
        )
        selected_kd = self._torch_default_table(
            self._select_randomization_default_columns(
                self._default_kd,
                columns,
                field=RESET_TERM_KD,
            )
        )
        return columns, selected_kp, selected_kd

    def write_actuator_gains(
        self,
        env_ids: torch.Tensor,
        actuator_ids: np.ndarray,
        kp: torch.Tensor,
        kd: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected per-environment actuator gains in the reset transaction."""
        ids = self._prepare_state_write(
            env_ids,
            capability="actuator-gain",
            term_name=term_name,
        )
        columns = self._validate_columns(
            actuator_ids,
            width=self._backend.num_actuators,
            capability="actuator IDs",
            term_name=term_name,
        )
        self._materialize_default_actuator_gains(term_name)
        gains_shape = (ids.size, columns.size)
        kp_values = self._validate_values(
            kp,
            shape=gains_shape,
            capability="actuator kp",
            term_name=term_name,
        )
        kd_values = self._validate_values(
            kd,
            shape=gains_shape,
            capability="actuator kd",
            term_name=term_name,
        )
        assert self._default_kp is not None
        assert self._default_kd is not None
        assert self._kp is not None
        assert self._kd is not None
        uninitialized = ids[~self._gain_dirty_mask[ids]]
        if uninitialized.size:
            self._kp[uninitialized] = self._randomization_default_rows(
                self._default_kp,
                uninitialized,
                field=RESET_TERM_KP,
            )
            self._kd[uninitialized] = self._randomization_default_rows(
                self._default_kd,
                uninitialized,
                field=RESET_TERM_KD,
            )
        if ids.size and columns.size:
            self._kp[ids[:, None], columns[None, :]] = kp_values
            self._kd[ids[:, None], columns[None, :]] = kd_values
        self._gain_dirty_mask[ids] = True
        self._dirty_mask[ids] = True

    def write_randomization_tensor_mask(
        self,
        field: str,
        reset_mask: torch.Tensor,
        columns: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Fail closed: reset DR has no mask commit boundary."""
        del field, reset_mask, columns, values
        raise NotImplementedError(
            f"EventManager term '{term_name}' reset randomization is unsupported by "
            "the fixed-width mask reset transaction"
        )

    def _materialize_tensor_randomization_default(self, field: str, term_name: str) -> torch.Tensor:
        """Build the immutable per-env device base for one DR field on the cold path."""
        device = self._packed_reset_device
        if device is None:
            raise RuntimeError("device tensor reset requires its selected device to be declared")
        if field == RESET_TERM_KP:
            self._materialize_default_actuator_gains(term_name)
            default = self._default_kp
        elif field == RESET_TERM_KD:
            self._materialize_default_actuator_gains(term_name)
            default = self._default_kd
        else:
            default = self._require_randomization_default(field, term_name)
        assert default is not None
        table = torch.as_tensor(np.array(default, dtype=np.float32, copy=True), device=device)
        if self._default_is_per_env(default, field=field):
            return table
        return table.unsqueeze(0).expand(self._num_envs, *table.shape).contiguous()

    def _tensor_randomization_columns(
        self, field: str, columns: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        """Cache validated DR column selectors as device index tensors."""
        key = (field, tuple(int(column) for column in columns))
        cached = self._tensor_dr_columns.get(key)
        if cached is None:
            cached = torch.as_tensor(columns, dtype=torch.int64, device=device)
            self._tensor_dr_columns[key] = cached
        return cached

    def _host_randomization_staged(self) -> bool:
        """Whether any term staged reset randomization through the NumPy path."""
        return bool(np.any(self._gain_dirty_mask)) or any(
            bool(np.any(mask)) for mask in self._randomization_dirty_masks.values()
        )

    def reset_to_default(self, reset_mask: torch.Tensor, *, term_name: str) -> None:
        """Stage backend default qpos/qvel for the active full-width mask."""
        self._require_active()
        active_mask = self._tensor_active_mask
        if active_mask is None:
            raise RuntimeError("fixed-width default reset requires an active mask transaction")
        if active_mask is not reset_mask:
            raise ValueError(
                f"EventManager term '{term_name}' attempted default reset outside the active mask"
            )
        if self.scene_layout is not None:
            raise NotImplementedError(
                "mask reset does not support mapped entity default-state restoration"
            )
        if not bool(reset_mask.any()):
            return
        self._requesting_terms.add(term_name)
        assert self._tensor_qpos is not None
        assert self._tensor_qvel is not None
        assert self._tensor_qpos_default is not None
        assert self._tensor_qvel_default is not None
        expanded = reset_mask[:, None]
        self._tensor_qpos.copy_(
            torch.where(
                expanded,
                self._tensor_qpos_default.expand_as(self._tensor_qpos),
                self._tensor_qpos,
            )
        )
        self._tensor_qvel.copy_(
            torch.where(
                expanded,
                self._tensor_qvel_default.expand_as(self._tensor_qvel),
                self._tensor_qvel,
            )
        )
        self._tensor_has_writes = True

    def write_joint_state(
        self,
        env_ids: torch.Tensor,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage selected joint position and velocity columns in the reset batch."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_joint_state")
        outside = ids[~self._active_mask[ids]]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted joint-state mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        assert self._qpos is not None
        assert self._qvel is not None

        pos_columns = self._validate_columns(
            qpos_indices,
            width=self._default_qpos.size,
            capability="qpos indices",
            term_name=term_name,
        )
        vel_columns = self._validate_columns(
            qvel_indices,
            width=self._default_qvel.size,
            capability="qvel indices",
            term_name=term_name,
        )
        if pos_columns.size != vel_columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' joint-state qpos/qvel index counts differ: "
                f"{pos_columns.size} != {vel_columns.size}"
            )
        positions = self._validate_values(
            position,
            shape=(ids.size, pos_columns.size),
            capability="joint position",
            term_name=term_name,
        )
        velocities = self._validate_values(
            velocity,
            shape=(ids.size, vel_columns.size),
            capability="joint velocity",
            term_name=term_name,
        )

        uninitialized = ids[~self._dirty_mask[ids]]
        if uninitialized.size:
            self._qpos[uninitialized] = self._default_qpos
            self._qvel[uninitialized] = self._default_qvel
        if ids.size and pos_columns.size:
            self._qpos[ids[:, None], pos_columns[None, :]] = positions
            self._qvel[ids[:, None], vel_columns[None, :]] = velocities
        self._dirty_mask[ids] = True

    def write_joint_state_mask(
        self,
        reset_mask: torch.Tensor,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        position: torch.Tensor,
        velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage full-width joint values for the active reset mask."""
        self._require_active()
        active_mask = self._tensor_active_mask
        if active_mask is None:
            raise RuntimeError("ManagerBased mask reset transaction is not active")
        if self._tensor_owner_subset:
            if bool((reset_mask & ~active_mask).any()):
                raise ValueError(
                    f"EventManager term '{term_name}' attempted joint-state mutation outside "
                    "the active mask"
                )
        elif active_mask is not reset_mask:
            raise ValueError(
                f"EventManager term '{term_name}' attempted joint-state mutation outside "
                "the active mask"
            )
        self._requesting_terms.add(term_name)
        assert self._tensor_qpos is not None
        assert self._tensor_qvel is not None
        pos_columns = self._validate_columns(
            qpos_indices,
            width=self._tensor_qpos.shape[1],
            capability="qpos indices",
            term_name=term_name,
        )
        vel_columns = self._validate_columns(
            qvel_indices,
            width=self._tensor_qvel.shape[1],
            capability="qvel indices",
            term_name=term_name,
        )
        if pos_columns.size != vel_columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' joint-state qpos/qvel index counts differ: "
                f"{pos_columns.size} != {vel_columns.size}"
            )
        positions = self._validate_tensor_values(
            position,
            shape=(self._num_envs, pos_columns.size),
            capability="joint position",
            term_name=term_name,
        )
        velocities = self._validate_tensor_values(
            velocity,
            shape=(self._num_envs, vel_columns.size),
            capability="joint velocity",
            term_name=term_name,
        )
        positions = positions.to(device=self._tensor_qpos.device)
        velocities = velocities.to(device=self._tensor_qvel.device)
        expanded = reset_mask[:, None]
        qpos_source = self._tensor_qpos[:, pos_columns]
        qvel_source = self._tensor_qvel[:, vel_columns]
        self._tensor_qpos[:, pos_columns] = torch.where(expanded, positions, qpos_source)
        self._tensor_qvel[:, vel_columns] = torch.where(expanded, velocities, qvel_source)
        self._tensor_has_writes = True

    def write_root_state(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        root_state: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage a community 13-D world-frame root state."""
        self._require_active()
        ids = self._validate_ids(env_ids, capability="write_root_state")
        values = self._validate_values(
            root_state,
            shape=(ids.size, 13),
            capability="root state",
            term_name=term_name,
        )
        self.write_root_pose(
            env_ids,
            layout,
            torch.from_numpy(values[:, :7]),
            term_name=term_name,
        )
        self.write_root_velocity(
            env_ids,
            layout,
            torch.from_numpy(values[:, 7:]),
            term_name=term_name,
        )

    def _validate_motion_tensor_payload(
        self,
        *,
        root_state: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        term_name: str,
    ) -> None:
        """Validate the fused motion payload through one device synchronization."""
        quat_norm = torch.linalg.vector_norm(root_state[:, 3:7], dim=-1)
        valid_quat = torch.isclose(
            quat_norm,
            torch.ones_like(quat_norm),
            rtol=1e-5,
            atol=1e-6,
        )
        valid = (
            torch.isfinite(root_state).all()
            & torch.isfinite(joint_position).all()
            & torch.isfinite(joint_velocity).all()
            & valid_quat.all()
        )
        # Avoid a Python bool conversion of the aggregate: scalar conversion is
        # itself synchronized and this fused reset path already has one device
        # reduction. Compare against a device-resident true scalar so the
        # success path remains non-synchronizing; invalid diagnostics below are
        # still allowed to synchronize and identify the offending payload.
        true_value = torch.ones_like(valid)
        if torch.equal(valid, true_value):
            return
        payloads = (
            ("root state", root_state),
            ("joint position", joint_position),
            ("joint velocity", joint_velocity),
        )
        for name, values in payloads:
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    f"EventManager term '{term_name}' tensor motion {name} contains NaN or Inf"
                )
        self._validate_tensor_root_quaternions(root_state[:, 3:7], term_name=term_name)

    def write_root_state_tensor(
        self,
        reset_mask: torch.Tensor,
        layout: BackendRootStateLayout,
        root_state: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage a full-width root-state payload selected by a bool mask."""
        mask = self._require_tensor_mask_write(
            reset_mask, capability="write_root_state_tensor_mask"
        )
        if root_state.shape != (self._num_envs, 13):
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask root state must have shape "
                f"{(self._num_envs, 13)}; got {tuple(root_state.shape)}"
            )
        if root_state.dtype != torch.float32 or not root_state.is_contiguous():
            raise TypeError(
                f"EventManager term '{term_name}' tensor mask root state must be contiguous float32"
            )
        if root_state.device != self._packed_reset_device:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask root state must live on "
                f"{self._packed_reset_device}; got {root_state.device}"
            )
        self._validate_tensor_root_payload(root_state, term_name=term_name)
        qpos = self._tensor_qpos
        qvel = self._tensor_qvel
        assert qpos is not None and qvel is not None
        qpos_columns_t, qvel_columns_t = self._tensor_root_columns_for_layout(
            layout, term_name=term_name
        )
        selected_qpos = torch.where(mask[:, None], root_state[:, :7], qpos[:, qpos_columns_t])
        selected_qvel = torch.where(mask[:, None], root_state[:, 7:13], qvel[:, qvel_columns_t])
        qpos[:, qpos_columns_t] = selected_qpos
        qvel[:, qvel_columns_t] = selected_qvel
        self._tensor_has_writes = True

    def _require_tensor_mask_write(
        self, reset_mask: torch.Tensor, *, capability: str
    ) -> torch.Tensor:
        if not self._tensor_active or self._tensor_active_mask is None:
            raise RuntimeError(
                f"ManagerBased tensor mask {capability} requires an active mask reset"
            )
        if reset_mask is not self._tensor_active_mask:
            raise ValueError(f"ManagerBased tensor mask {capability} must use the transaction mask")
        if self._tensor_owner_subset:
            raise RuntimeError(
                f"ManagerBased tensor mask {capability} does not support owner subsets"
            )
        self._requesting_terms.add(capability)
        return self._tensor_active_mask

    def write_motion_state_tensor(
        self,
        reset_mask: torch.Tensor,
        layout: BackendRootStateLayout,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        root_state: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage one full-width motion root and joint payload by bool mask."""
        mask = self._require_tensor_mask_write(
            reset_mask, capability="write_motion_state_tensor_mask"
        )
        qpos_columns, qvel_columns = self._tensor_joint_columns_for_indices(
            qpos_indices, qvel_indices, term_name=term_name
        )
        width = qpos_columns.numel()
        expected = (self._num_envs, width)
        if root_state.shape != (self._num_envs, 13):
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion root state must "
                f"have shape {(self._num_envs, 13)}; got {tuple(root_state.shape)}"
            )
        if joint_position.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion joint position must "
                f"have shape {expected}; got {tuple(joint_position.shape)}"
            )
        if joint_velocity.shape != joint_position.shape:
            raise ValueError(
                f"EventManager term '{term_name}' tensor mask motion joint velocity must "
                f"have shape {tuple(joint_position.shape)}"
            )
        for name, values in (
            ("root state", root_state),
            ("joint position", joint_position),
            ("joint velocity", joint_velocity),
        ):
            if values.dtype != torch.float32 or not values.is_contiguous():
                raise TypeError(
                    f"EventManager term '{term_name}' tensor mask motion {name} must "
                    "be contiguous float32"
                )
        self._validate_motion_tensor_payload(
            root_state=root_state,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            term_name=term_name,
        )
        qpos = self._tensor_qpos
        qvel = self._tensor_qvel
        assert qpos is not None and qvel is not None
        root_qpos_columns, root_qvel_columns = self._tensor_root_columns_for_layout(
            layout, term_name=term_name
        )
        next_qpos = qpos.clone()
        next_qvel = qvel.clone()
        next_qpos[:, root_qpos_columns] = torch.where(
            mask[:, None], root_state[:, :7], qpos[:, root_qpos_columns]
        )
        next_qvel[:, root_qvel_columns] = torch.where(
            mask[:, None], root_state[:, 7:13], qvel[:, root_qvel_columns]
        )
        next_qpos[:, qpos_columns] = torch.where(
            mask[:, None], joint_position, qpos[:, qpos_columns]
        )
        next_qvel[:, qvel_columns] = torch.where(
            mask[:, None], joint_velocity, qvel[:, qvel_columns]
        )
        qpos.copy_(next_qpos)
        qvel.copy_(next_qvel)
        self._tensor_has_writes = True

    def write_root_pose(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        pose: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage world position and wxyz orientation for one floating root."""
        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)
        positions = self._validate_values(
            pose,
            shape=(ids.size, 7),
            capability="root pose",
            term_name=term_name,
        )
        self._validate_quaternions(positions[:, 3:7], term_name=term_name)
        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)
        assert self._qpos is not None
        if ids.size:
            self._qpos[ids[:, None], qpos_columns[None, :]] = positions
        self._dirty_mask[ids] = True

    def write_root_velocity(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        velocity_w: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Stage world-frame root velocity in generalized qvel columns.

        The public generalized-state contract stores free-root linear velocity
        in world coordinates and angular velocity in root-body coordinates.
        The conversion uses the pose already staged in this transaction.
        """
        ids = self._prepare_state_write(env_ids, capability="root-velocity", term_name=term_name)
        velocities = self._validate_values(
            velocity_w,
            shape=(ids.size, 6),
            capability="root velocity",
            term_name=term_name,
        )
        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)
        assert self._qpos is not None
        assert self._qvel is not None
        if ids.size:
            quaternions = self._qpos[
                ids[:, None],
                qpos_columns[None, 3:7],
            ]
            self._validate_quaternions(quaternions, term_name=term_name)
            encoded_velocity = np.array(velocities, copy=True)
            encoded_velocity[:, 3:6] = np_quat_apply_inverse(
                quaternions,
                velocities[:, 3:6],
            )
            self._qvel[ids[:, None], qvel_columns[None, :]] = encoded_velocity
        self._dirty_mask[ids] = True

    def read_root_pose(
        self,
        env_ids: torch.Tensor,
        layout: BackendRootStateLayout,
        *,
        term_name: str,
    ) -> np.ndarray:
        """Read the world position and wxyz orientation staged for one floating root.

        Returns a detached ``(len(env_ids), 7)`` copy of the pose currently
        staged in this transaction. Rows no term has written yet are first
        initialized to the backend default pose, so a later reset term can
        build on an earlier term's root placement without re-deriving it.
        """
        ids = self._prepare_state_write(env_ids, capability="root-pose", term_name=term_name)
        qpos_columns, _ = self._validate_root_layout(layout, term_name=term_name)
        assert self._qpos is not None
        return np.array(self._qpos[ids[:, None], qpos_columns[None, :]], copy=True)

    def bind_mocap_pose(self, body_name: str) -> BackendMocapPoseBinding:
        """Resolve a mocap body once, without exposing a native backend handle."""
        if body_name not in self._mocap_bindings:
            if self._active:
                raise RuntimeError("Mocap bodies must be bound before reset starts")
            binding = self._backend.bind_mocap_pose(body_name)
            if binding.num_envs != self._num_envs or binding.body_name != body_name:
                raise ValueError("Backend mocap binding does not match the requested body/batch")
            self._mocap_bindings[body_name] = binding
            self._mocap_values[body_name] = np.empty((self._num_envs, 7))
            self._mocap_masks[body_name] = np.zeros(self._num_envs, dtype=np.bool_)
        return self._mocap_bindings[body_name]

    def read_mocap_pose(self, body_name: str) -> np.ndarray:
        """Read current poses with this transaction's pending rows overlaid."""
        poses = self._mocap_bindings[body_name].read()
        mask = self._mocap_masks[body_name]
        poses[mask] = self._mocap_values[body_name][mask]
        return poses

    def write_mocap_pose(
        self, body_name: str, env_ids: torch.Tensor, poses: np.ndarray, *, term_name: str
    ) -> None:
        """Stage poses; upload only after the ordinary reset state has committed."""
        self._require_active()
        if body_name not in self._mocap_bindings:
            raise RuntimeError(f"Mocap body {body_name!r} must be bound before writing")
        ids = self._validate_ids(env_ids, capability="mocap pose")
        if np.any(~self._active_mask[ids]):
            raise ValueError(f"{term_name}: mocap mutation outside the active reset")
        values = self._validate_values(
            poses, shape=(ids.size, 7), capability="mocap pose", term_name=term_name
        )
        self._validate_quaternions(values[:, 3:], term_name=term_name)
        self._mocap_values[body_name][ids] = values
        self._mocap_masks[body_name][ids] = True
        self._requesting_terms.add(term_name)

    def commit(self) -> dict | None:
        """Commit all staged rows through one public backend call."""
        self._require_active()
        if self.scene_layout is not None:
            try:
                self._commit_entities()
                if np.any(self._dirty_mask):
                    raise NotImplementedError(
                        "mapped scene DR writes need a public entity transaction"
                    )
                return None
            finally:
                self._finish()
        dirty_ids = np.flatnonzero(self._dirty_mask).astype(np.int32, copy=False)
        mocap_dirty = any(np.any(mask) for mask in self._mocap_masks.values())
        self._last_commit_had_writes = bool(dirty_ids.size) or mocap_dirty
        try:
            if dirty_ids.size == 0:
                self._commit_mocap_poses()
                return None
            assert self._qpos is not None
            assert self._qvel is not None
            randomization = self._build_randomization_payload(dirty_ids)
            try:
                set_state_t0 = time.perf_counter()
                result = self._backend.set_state(
                    dirty_ids,
                    self._qpos[dirty_ids],
                    self._qvel[dirty_ids],
                    randomization=randomization,
                )
                self._record_committed_payload(dirty_ids, randomization)
                if self._startup_randomization_fields:
                    self._promote_startup_randomization_defaults(dirty_ids)
                    self._startup_randomization_fields.clear()
                self._commit_mocap_poses()
                timing: dict[str, float] = {
                    "dr_reset_set_state_ms": (time.perf_counter() - set_state_t0) * 1000.0
                }
                if isinstance(result, dict):
                    backend_timing = result.get("timing")
                    if isinstance(backend_timing, dict):
                        timing.update(backend_timing)
                self._last_set_state_timing_ms = timing
                return cast(dict | None, result)
            except (AttributeError, NotImplementedError) as exc:
                terms = ", ".join(sorted(self._requesting_terms))
                raise NotImplementedError(
                    "EventManager reset-state capability 'SimBackend.set_state' is unavailable "
                    f"for term(s) [{terms}] on backend '{self._backend.backend_type}': {exc}"
                ) from exc
        finally:
            self._finish()

    def commit_device_tensor(self) -> dict | None:
        """Commit the fixed-width reset through the public mask API."""
        self._require_active()
        self._last_commit_had_writes = self._tensor_has_writes
        if self._tensor_active_mask is None:
            raise RuntimeError("fixed-width tensor reset is not active")
        try:
            if not self._tensor_has_writes:
                return None
            if self.scene_layout is not None:
                raise NotImplementedError(
                    "fixed-width tensor reset supports scalar qpos/qvel rows only; mapped "
                    "entity writes remain an explicit migration boundary"
                )
            capabilities = self._backend.get_tensor_capabilities()
            if capabilities.execution is not TensorExecution.DEVICE_RESIDENT:
                raise NotImplementedError(
                    "fixed-width reset requires DEVICE_RESIDENT tensor execution; "
                    f"backend '{self._backend.backend_type}' declares {capabilities.execution}"
                )
            if not capabilities.selected_reset:
                raise NotImplementedError(
                    "fixed-width reset requires the backend's declared selected_reset "
                    "tensor capability"
                )
            if not capabilities.mask_selected_reset:
                raise NotImplementedError(
                    "fixed-width reset requires the backend's declared "
                    "'mask_selected_reset' capability"
                )
            if self._mask_full_refresh and not capabilities.mask_selected_reset_full_refresh:
                raise NotImplementedError(
                    "fixed-width full refresh requires the backend's declared "
                    "'mask_selected_reset_full_refresh' capability"
                )
            assert self._tensor_qpos is not None
            assert self._tensor_qvel is not None
            reset_mask = self._tensor_active_mask
            try:
                started = time.perf_counter()
                set_state_mask = self._backend.set_state_tensor_mask
                if self._mask_full_refresh:
                    result = set_state_mask(
                        reset_mask,
                        self._tensor_qpos,
                        self._tensor_qvel,
                        full_refresh=True,
                    )
                else:
                    result = set_state_mask(
                        reset_mask,
                        self._tensor_qpos,
                        self._tensor_qvel,
                    )
                self._last_set_state_timing_ms = {
                    "dr_reset_set_state_ms": (time.perf_counter() - started) * 1000.0
                }
                if isinstance(result, dict):
                    backend_timing = result.get("timing")
                    if isinstance(backend_timing, dict):
                        self._last_set_state_timing_ms.update(backend_timing)
                return cast(dict | None, result)
            except (
                AttributeError,
                NotImplementedError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                terms = ", ".join(sorted(self._requesting_terms))
                raise NotImplementedError(
                    "EventManager reset-state capability "
                    "'SimBackend.set_state_tensor_mask' is unavailable for term(s) "
                    f"[{terms}] on backend '{self._backend.backend_type}': {exc}"
                ) from exc
        finally:
            self._finish()

    def _packed_reset_widths_match(self) -> bool:
        try:
            qpos_width, qvel_width = self._packed_reset_public_widths()
        except (AttributeError, NotImplementedError):
            return False
        if not isinstance(qpos_width, (int, np.integer)) or not isinstance(
            qvel_width, (int, np.integer)
        ):
            return False
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        return bool(qpos_width == self._default_qpos.size and qvel_width == self._default_qvel.size)

    def _packed_reset_public_widths(self) -> tuple[int, int]:
        widths = self._backend.get_public_state_widths()
        return int(widths.nq), int(widths.nv)

    def _commit_mocap_poses(self) -> None:
        for name, mask in self._mocap_masks.items():
            ids = np.flatnonzero(mask).astype(np.int32, copy=False)
            if ids.size:
                self._mocap_bindings[name].write(ids, self._mocap_values[name][ids])

    def abort(self) -> None:
        """Discard staged rows without touching the backend."""
        if self._active or self._tensor_active:
            self._finish()

    def _materialize_default_state(self, term_name: str) -> None:
        if self._default_qpos is not None:
            return
        qpos = self._selected_default_qpos
        if qpos is None:
            try:
                qpos = self._backend.get_default_qpos()
            except (AttributeError, NotImplementedError) as exc:
                raise self._capability_error(term_name, "default qpos", exc) from exc
        try:
            qvel = self._backend.get_init_qvel()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "initial qvel", exc) from exc

        default_qpos = self._validate_state_vector(qpos, "default qpos", term_name)
        default_qvel = self._validate_state_vector(qvel, "initial qvel", term_name)
        self._default_qpos = default_qpos
        self._default_qvel = default_qvel
        self._qpos = np.empty((self._num_envs, default_qpos.size), dtype=default_qpos.dtype)
        self._qvel = np.empty((self._num_envs, default_qvel.size), dtype=default_qvel.dtype)

    def _materialize_tensor_default_state(self, term_name: str) -> None:
        """Materialize immutable default state and full-width device staging."""
        public_widths = self._backend.get_public_state_widths()
        if (
            self._tensor_qpos is not None
            and self._tensor_qpos.shape[1] == int(public_widths.nq)
            and self._tensor_qvel is not None
            and self._tensor_qvel.shape[1] == int(public_widths.nv)
        ):
            return
        self._materialize_default_state(term_name)
        device = self._packed_reset_device
        if device is None:
            raise RuntimeError("device tensor reset requires its selected device to be declared")
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        if self._default_qpos.size != int(public_widths.nq):
            self._default_qpos = self._validate_state_vector(
                self._backend.get_default_qpos(), "default qpos", term_name
            )
        if self._default_qvel.size != int(public_widths.nv):
            self._default_qvel = self._validate_state_vector(
                self._backend.get_init_qvel(), "initial qvel", term_name
            )
        qpos_default = torch.from_numpy(
            np.array(self._default_qpos, dtype=np.float32, copy=True)
        ).to(device=device)
        qvel_default = torch.from_numpy(
            np.array(self._default_qvel, dtype=np.float32, copy=True)
        ).to(device=device)
        self._tensor_qpos = qpos_default.repeat(self._num_envs, 1).clone()
        self._tensor_qvel = qvel_default.repeat(self._num_envs, 1).clone()
        self._tensor_qpos_default = qpos_default
        self._tensor_qvel_default = qvel_default

    def _reset_tensor_staging_mask(self, reset_mask: torch.Tensor) -> None:
        """Restore full-width default staging for rows selected by this reset."""
        qpos = self._tensor_qpos
        qvel = self._tensor_qvel
        qpos_default = self._tensor_qpos_default
        qvel_default = self._tensor_qvel_default
        assert qpos is not None and qvel is not None
        assert qpos_default is not None and qvel_default is not None
        expanded = reset_mask[:, None]
        qpos.copy_(torch.where(expanded, qpos_default.expand_as(qpos), qpos))
        qvel.copy_(torch.where(expanded, qvel_default.expand_as(qvel), qvel))

    def _validate_tensor_root_payload(self, root_state: torch.Tensor, *, term_name: str) -> None:
        """Validate finite values and quaternion norms through one device comparison."""
        quat_norm = torch.linalg.vector_norm(root_state[:, 3:7], dim=-1)
        valid_quat = torch.isclose(
            quat_norm,
            torch.ones_like(quat_norm),
            rtol=1e-5,
            atol=1e-6,
        )
        valid = torch.isfinite(root_state).all() & valid_quat.all()
        if torch.equal(valid, torch.ones_like(valid)):
            return
        if not bool(torch.isfinite(root_state).all()):
            raise ValueError(
                f"EventManager term '{term_name}' tensor root state contains NaN or Inf"
            )
        self._validate_tensor_root_quaternions(root_state[:, 3:7], term_name=term_name)

    def _validate_tensor_root_quaternions(self, values: torch.Tensor, *, term_name: str) -> None:
        norms = torch.linalg.vector_norm(values, dim=-1)
        if not bool(torch.isclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6).all()):
            raise ValueError(
                f"EventManager term '{term_name}' tensor root quaternion must be unit length"
            )

    def _materialize_default_actuator_gains(self, term_name: str) -> None:
        if self._default_kp is not None:
            return
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, "actuator gain randomization", exc) from exc
        required = frozenset((RESET_TERM_KP, RESET_TERM_KD))
        unsupported = capabilities.get_unsupported_reset_terms(required)
        if unsupported:
            detail = ", ".join(sorted(unsupported))
            raise self._capability_error(
                term_name,
                "actuator gain randomization",
                NotImplementedError(f"unsupported reset payload fields: {detail}"),
            )
        default_kp = self._fetch_reset_term_default(
            RESET_TERM_KP,
            expected_tail=(),
            term_name=term_name,
        )
        default_kd = self._fetch_reset_term_default(
            RESET_TERM_KD,
            expected_tail=(),
            term_name=term_name,
        )
        for name, default in ((RESET_TERM_KP, default_kp), (RESET_TERM_KD, default_kd)):
            width = self._randomization_default_width(default, field=name)
            if width != self._backend.num_actuators:
                raise ValueError(
                    f"EventManager term '{term_name}' default actuator {name} on backend "
                    f"'{self._backend.backend_type}' has model width {width}; expected "
                    f"{self._backend.num_actuators}"
                )
        self._default_kp = default_kp
        self._default_kd = default_kd
        self._kp = np.empty(
            (self._num_envs, self._backend.num_actuators),
            dtype=default_kp.dtype,
        )
        self._kd = np.empty(
            (self._num_envs, self._backend.num_actuators),
            dtype=default_kd.dtype,
        )

    def _materialize_randomization_default(
        self,
        field: str,
        *,
        expected_tail: tuple[int, ...],
        term_name: str,
    ) -> np.ndarray:
        cached = self._randomization_defaults.get(field)
        if cached is not None:
            return cached
        try:
            capabilities = self._backend.get_dr_capabilities()
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"{field} randomization", exc) from exc
        unsupported = capabilities.get_unsupported_reset_terms(frozenset((field,)))
        if unsupported:
            raise self._capability_error(
                term_name,
                f"{field} randomization",
                NotImplementedError(f"unsupported reset payload field: {field}"),
            )
        default = self._fetch_reset_term_default(
            field,
            expected_tail=expected_tail,
            term_name=term_name,
        )
        self._randomization_defaults[field] = default
        self._randomization_values[field] = np.empty(
            (self._num_envs, *self._canonical_default_shape(default, field=field)),
            dtype=default.dtype,
        )
        self._randomization_dirty_masks[field] = np.zeros(self._num_envs, dtype=np.bool_)
        return default

    def _fetch_reset_term_default(
        self,
        field: str,
        *,
        expected_tail: tuple[int, ...],
        term_name: str,
    ) -> np.ndarray:
        try:
            value = self._backend.get_reset_term_default(field)
        except (AttributeError, NotImplementedError) as exc:
            raise self._capability_error(term_name, f"default {field}", exc) from exc
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        canonical_ndim = 1 + len(expected_tail)
        canonical_shape = value.ndim == canonical_ndim
        per_env_shape = value.ndim == canonical_ndim + 1 and value.shape[0] == self._num_envs
        if field == RESET_TERM_GRAVITY:
            canonical_shape = value.shape == (3,)
            per_env_shape = value.shape == (self._num_envs, 3)
        if not canonical_shape and not per_env_shape:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected "
                f"a canonical {canonical_ndim}-D table or a per-environment "
                f"({self._num_envs}, *canonical) table"
            )
        tail_slice = 2 if per_env_shape else 1
        if value.shape[tail_slice:] != expected_tail:
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected tail "
                f"{expected_tail}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability 'default {field}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        default = np.array(value, copy=True)
        default.setflags(write=False)
        return default

    def _require_randomization_default(self, field: str, term_name: str) -> np.ndarray:
        try:
            return self._randomization_defaults[field]
        except KeyError as exc:
            raise RuntimeError(
                f"EventManager term '{term_name}' must bind reset field '{field}' "
                "during manager construction before writing it"
            ) from exc

    def _default_is_per_env(
        self,
        default: np.ndarray,
        *,
        field: str,
    ) -> bool:
        canonical_ndim = 1 + len(_randomization_term_tail(field))
        if default.ndim == canonical_ndim:
            return False
        if default.ndim == canonical_ndim + 1 and default.shape[0] == self._num_envs:
            return True
        raise RuntimeError(
            f"Reset transaction cached an invalid '{field}' default table with shape "
            f"{default.shape}"
        )

    def _canonical_default_shape(
        self,
        default: np.ndarray,
        *,
        field: str,
    ) -> tuple[int, ...]:
        shape = (
            default.shape[1:] if self._default_is_per_env(default, field=field) else default.shape
        )
        return tuple(int(value) for value in shape)

    def _randomization_default_width(
        self,
        default: np.ndarray,
        *,
        field: str,
    ) -> int:
        axis = 1 if self._default_is_per_env(default, field=field) else 0
        return int(default.shape[axis])

    def _select_randomization_default_columns(
        self,
        default: np.ndarray,
        columns: np.ndarray,
        *,
        field: str,
    ) -> np.ndarray:
        if self._default_is_per_env(default, field=field):
            return default[:, columns]
        return default[columns]

    def _randomization_default_rows(
        self,
        default: np.ndarray,
        env_ids: np.ndarray,
        *,
        field: str,
    ) -> np.ndarray:
        return default[env_ids] if self._default_is_per_env(default, field=field) else default

    def record_startup_randomization(
        self,
        field: str,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: torch.Tensor,
        *,
        term_name: str,
    ) -> None:
        """Mark staged startup rows to become the reset baseline after commit.

        The backend remains authoritative for committed state. This local cache
        prevents a later mask reset from rebuilding unwritten model
        columns from immutable construction defaults and clobbering the startup
        values. The promotion is deferred until the public backend commit
        succeeds, preserving the reset transaction's all-or-nothing contract.
        """
        if field not in self._randomization_defaults:
            # Validation and immutable-default binding are owned by the
            # corresponding write performed by the startup event.
            raise RuntimeError(
                f"EventManager term '{term_name}' must bind reset field '{field}' "
                "before recording startup randomization"
            )
        ids = self._validate_ids(env_ids, capability="startup randomization")
        if ids.size != self._num_envs or not np.array_equal(ids, np.arange(self._num_envs)):
            raise ValueError(
                f"EventManager term '{term_name}' startup randomization field '{field}' "
                f"requires all {self._num_envs} environment rows; got {ids.tolist()}"
            )
        tail = _randomization_term_tail(field)
        self._write_selected_randomization(
            field,
            env_ids,
            column_ids,
            values,
            value_tail=tail,
            term_name=term_name,
        )
        self._startup_randomization_fields.add(field)

    def _promote_startup_randomization_defaults(self, dirty_ids: np.ndarray) -> None:
        """Promote successfully committed full-width startup tables to baselines."""
        if not self._startup_randomization_fields:
            return
        if dirty_ids.size != self._num_envs or not np.array_equal(
            dirty_ids, np.arange(self._num_envs)
        ):
            raise RuntimeError(
                "startup model-field randomization requires one full-width commit; got rows "
                f"{dirty_ids.tolist()} for {self._num_envs} environments"
            )
        for field in tuple(self._startup_randomization_fields):
            committed = self._committed_randomization.get(field)
            mask = self._committed_randomization_masks.get(field)
            if committed is None or mask is None or not bool(mask.all()):
                raise RuntimeError(
                    f"startup model-field randomization did not commit a complete '{field}' table"
                )
            promoted = np.array(committed, copy=True)
            promoted.setflags(write=False)
            self._randomization_defaults[field] = promoted

    def _readonly_binding(
        self,
        columns: np.ndarray,
        selected_default: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _readonly_array(columns), _readonly_array(selected_default)

    def _write_selected_randomization(
        self,
        field: str,
        env_ids: torch.Tensor,
        column_ids: np.ndarray,
        values: np.ndarray | torch.Tensor,
        *,
        value_tail: tuple[int, ...],
        term_name: str,
    ) -> None:
        ids = self._prepare_state_write(
            env_ids,
            capability=field,
            term_name=term_name,
        )
        default = self._require_randomization_default(field, term_name)
        columns = self._validate_columns(
            column_ids,
            width=self._randomization_default_width(default, field=field),
            capability=f"{field} column IDs",
            term_name=term_name,
        )
        selected = self._validate_values(
            values,
            shape=(ids.size, columns.size, *value_tail),
            capability=field,
            term_name=term_name,
        )
        buffer = self._randomization_values[field]
        mask = self._randomization_dirty_masks[field]
        uninitialized = ids[~mask[ids]]
        if uninitialized.size:
            buffer[uninitialized] = self._randomization_default_rows(
                default,
                uninitialized,
                field=field,
            )
        if ids.size and columns.size:
            buffer[ids[:, None], columns[None, :]] = selected
        mask[ids] = True
        self._dirty_mask[ids] = True

    def _build_randomization_payload(
        self,
        dirty_ids: np.ndarray,
    ) -> ResetRandomizationPayload | None:
        payload = ResetRandomizationPayload()
        for field in (
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_DOF_DAMPING,
            RESET_TERM_DOF_FRICTIONLOSS,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_GEOM_SIZE,
            RESET_TERM_GEOM_SOLREF,
            RESET_TERM_GEOM_SOLIMP,
            RESET_TERM_GRAVITY,
        ):
            mask = self._randomization_dirty_masks.get(field)
            if mask is None or not np.any(mask):
                continue
            self._fill_sparse_randomization_rows(field, mask, dirty_ids)
            setattr(
                payload,
                field,
                np.array(self._randomization_values[field][dirty_ids], copy=True),
            )

        gain_ids = np.flatnonzero(self._gain_dirty_mask).astype(np.int32, copy=False)
        if gain_ids.size:
            self._fill_sparse_gain_rows(dirty_ids)
            assert self._kp is not None
            assert self._kd is not None
            payload.kp = np.array(self._kp[dirty_ids], copy=True)
            payload.kd = np.array(self._kd[dirty_ids], copy=True)
        return None if payload.is_empty() else payload

    def _fill_sparse_randomization_rows(
        self,
        field: str,
        mask: np.ndarray,
        dirty_ids: np.ndarray,
    ) -> None:
        """Fill dirty rows the current reset did not rewrite from committed values.

        Terms gated by ``min_step_count_between_reset`` intentionally skip envs
        whose last randomized value must persist; the last committed payload
        re-supplies those rows so one dense ``SimBackend.set_state`` call can
        represent them. Rows with no committed history still fail closed.
        """
        missing = dirty_ids[~mask[dirty_ids]]
        if not missing.size:
            return
        cached = self._committed_randomization.get(field)
        cached_mask = self._committed_randomization_masks.get(field)
        if cached is not None and cached_mask is not None:
            uncommitted = missing[~cached_mask[missing]]
        else:
            uncommitted = missing
        if uncommitted.size:
            terms = ", ".join(sorted(self._requesting_terms))
            raise RuntimeError(
                f"EventManager reset {field} payload cannot represent sparse rows in one "
                f"SimBackend.set_state call for term(s) [{terms}] on backend "
                f"'{self._backend.backend_type}'; missing env IDs {uncommitted.tolist()}"
            )
        assert cached is not None
        self._randomization_values[field][missing] = cached[missing]

    def _fill_sparse_gain_rows(self, dirty_ids: np.ndarray) -> None:
        """Fill gain rows the current reset did not rewrite from committed values."""
        missing = dirty_ids[~self._gain_dirty_mask[dirty_ids]]
        if not missing.size:
            return
        uncommitted = missing[~self._committed_gain_mask[missing]]
        if uncommitted.size or self._committed_kp is None or self._committed_kd is None:
            terms = ", ".join(sorted(self._requesting_terms))
            raise RuntimeError(
                "EventManager reset actuator gains payload cannot represent sparse rows in "
                f"one SimBackend.set_state call for term(s) [{terms}] on backend "
                f"'{self._backend.backend_type}'; missing env IDs {uncommitted.tolist()}"
            )
        assert self._kp is not None
        assert self._kd is not None
        self._kp[missing] = self._committed_kp[missing]
        self._kd[missing] = self._committed_kd[missing]

    def _record_committed_payload(
        self,
        dirty_ids: np.ndarray,
        payload: ResetRandomizationPayload | None,
    ) -> None:
        """Cache the last committed per-env field values for sparse-row fill."""
        if payload is None:
            return
        for field in (
            RESET_TERM_BODY_INERTIA,
            RESET_TERM_BODY_MASS,
            RESET_TERM_BODY_IPOS,
            RESET_TERM_DOF_ARMATURE,
            RESET_TERM_DOF_DAMPING,
            RESET_TERM_DOF_FRICTIONLOSS,
            RESET_TERM_GEOM_FRICTION,
            RESET_TERM_GEOM_SIZE,
            RESET_TERM_GEOM_SOLREF,
            RESET_TERM_GEOM_SOLIMP,
            RESET_TERM_GRAVITY,
        ):
            values = getattr(payload, field)
            if values is None:
                continue
            cache = self._committed_randomization.get(field)
            if cache is None:
                cache = np.empty(
                    (self._num_envs, *values.shape[1:]),
                    dtype=values.dtype,
                )
                self._committed_randomization[field] = cache
                self._committed_randomization_masks[field] = np.zeros(
                    self._num_envs, dtype=np.bool_
                )
            cache[dirty_ids] = values
            self._committed_randomization_masks[field][dirty_ids] = True
            # A host commit is newer than the device-committed baseline; force
            # the next device-staged reset of this field to rebuild from the
            # materialized defaults rather than stale device rows.
            self._tensor_dr_committed.pop(field, None)
        if payload.kp is not None and payload.kd is not None:
            if self._committed_kp is None or self._committed_kd is None:
                self._committed_kp = np.empty(
                    (self._num_envs, payload.kp.shape[1]), dtype=payload.kp.dtype
                )
                self._committed_kd = np.empty(
                    (self._num_envs, payload.kd.shape[1]), dtype=payload.kd.dtype
                )
            self._committed_kp[dirty_ids] = payload.kp
            self._committed_kd[dirty_ids] = payload.kd
            self._committed_gain_mask[dirty_ids] = True
            self._tensor_dr_committed.pop(RESET_TERM_KP, None)
            self._tensor_dr_committed.pop(RESET_TERM_KD, None)

    def _prepare_state_write(
        self,
        env_ids: torch.Tensor,
        *,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        self._require_active()
        if self._tensor_active:
            return self._prepare_tensor_randomization_write(
                env_ids, capability=capability, term_name=term_name
            )
        ids = self._validate_ids(env_ids, capability=f"write_{capability}")
        outside = ids[~self._active_mask[ids]]
        if outside.size:
            raise ValueError(
                f"EventManager term '{term_name}' attempted {capability} mutation outside "
                f"the active reset: {outside.tolist()}"
            )
        self._requesting_terms.add(term_name)
        self._materialize_default_state(term_name)
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        assert self._qpos is not None
        assert self._qvel is not None
        uninitialized = ids[~self._dirty_mask[ids]]
        if uninitialized.size:
            self._qpos[uninitialized] = self._default_qpos
            self._qvel[uninitialized] = self._default_qvel
        return ids

    def _prepare_tensor_randomization_write(
        self,
        env_ids: torch.Tensor,
        *,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        """Fail closed because the mask reset contract does not stage DR rows."""
        del env_ids, capability
        raise NotImplementedError(
            f"EventManager term '{term_name}' reset randomization is not supported by "
            "the fixed-width mask reset transaction"
        )

    def _validate_root_layout(
        self,
        layout: BackendRootStateLayout,
        *,
        term_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(layout, BackendRootStateLayout):
            raise TypeError(
                f"EventManager term '{term_name}' root-state layout must be "
                f"BackendRootStateLayout, got {type(layout).__name__}"
            )
        assert self._default_qpos is not None
        assert self._default_qvel is not None
        qpos_columns = self._validate_columns(
            np.asarray(layout.qpos_indices, dtype=np.intp),
            width=self._default_qpos.size,
            capability="root qpos indices",
            term_name=term_name,
        )
        qvel_columns = self._validate_columns(
            np.asarray(layout.qvel_indices, dtype=np.intp),
            width=self._default_qvel.size,
            capability="root qvel indices",
            term_name=term_name,
        )
        return qpos_columns, qvel_columns

    def _tensor_root_columns_for_layout(
        self, layout: BackendRootStateLayout, *, term_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos_columns, qvel_columns = self._validate_root_layout(layout, term_name=term_name)
        key = (tuple(int(i) for i in qpos_columns), tuple(int(i) for i in qvel_columns))
        cached = self._tensor_root_columns
        device = self._packed_reset_device
        assert device is not None
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2], cached[3]
        qpos_tensor = torch.as_tensor(qpos_columns, dtype=torch.int64, device=device)
        qvel_tensor = torch.as_tensor(qvel_columns, dtype=torch.int64, device=device)
        self._tensor_root_columns = (*key, qpos_tensor, qvel_tensor)
        return qpos_tensor, qvel_tensor

    def _tensor_joint_columns_for_indices(
        self,
        qpos_indices: np.ndarray,
        qvel_indices: np.ndarray,
        *,
        term_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._default_qpos is not None and self._default_qvel is not None
        qpos_columns = self._validate_columns(
            np.asarray(qpos_indices, dtype=np.intp),
            width=self._default_qpos.size,
            capability="tensor joint qpos indices",
            term_name=term_name,
        )
        qvel_columns = self._validate_columns(
            np.asarray(qvel_indices, dtype=np.intp),
            width=self._default_qvel.size,
            capability="tensor joint qvel indices",
            term_name=term_name,
        )
        key = (
            tuple(int(index) for index in qpos_columns),
            tuple(int(index) for index in qvel_columns),
        )
        cached = self._tensor_joint_columns
        device = self._packed_reset_device
        assert device is not None
        if cached is not None and (cached[0], cached[1]) == key:
            return cached[2], cached[3]
        qpos_tensor = torch.as_tensor(qpos_columns, dtype=torch.int64, device=device)
        qvel_tensor = torch.as_tensor(qvel_columns, dtype=torch.int64, device=device)
        self._tensor_joint_columns = (*key, qpos_tensor, qvel_tensor)
        return qpos_tensor, qvel_tensor

    def _validate_quaternions(self, values: np.ndarray, *, term_name: str) -> None:
        norms = np.linalg.norm(values, axis=1)
        invalid = ~np.isclose(norms, 1.0, rtol=1e-5, atol=1e-6)
        if np.any(invalid):
            raise ValueError(
                f"EventManager term '{term_name}' root quaternion must be unit length; "
                f"norms={norms[invalid].tolist()}"
            )

    def _validate_state_vector(
        self,
        value: np.ndarray,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must return np.ndarray, got "
                f"{type(value).__name__}"
            )
        if value.ndim != 1:
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned shape {value.shape}; expected 1-D"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' must be floating, got {value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned NaN or Inf"
            )
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result

    def _validate_gain_vector(
        self,
        value: np.ndarray,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        result = self._validate_state_vector(value, capability, term_name)
        expected = (self._backend.num_actuators,)
        if result.shape != expected:
            raise ValueError(
                f"EventManager term '{term_name}' capability '{capability}' on backend "
                f"'{self._backend.backend_type}' returned shape {result.shape}; expected {expected}"
            )
        return result

    def _validate_ids(self, env_ids: torch.Tensor, *, capability: str) -> np.ndarray:
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be torch.Tensor, "
                f"got {type(env_ids).__name__}"
            )
        if env_ids.ndim != 1 or env_ids.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                f"ManagerBased reset-state {capability} env_ids must be a 1-D integer "
                f"tensor, got shape={tuple(env_ids.shape)}, dtype={env_ids.dtype}"
            )
        rows = env_ids.to(dtype=torch.int64)
        if rows.numel() and not bool(((rows >= 0) & (rows < self._num_envs)).all()):
            raise IndexError(
                f"ManagerBased reset-state {capability} env_ids out of range for "
                f"{self._num_envs} environments: {rows.tolist()}"
            )
        ids = rows.detach().cpu().numpy().astype(np.int32, copy=False)
        # Duplicate check via bincount instead of np.unique: identical semantics
        # (ids are already range-checked above) but avoids the sort — ~30x faster
        # at num_envs=4096 and ~4x at typical partial-reset widths. This runs on
        # every reset-state write (~6x per env step), so the sort cost was
        # measurable in the collector host phase (issue #1352).
        if ids.size > 1 and np.bincount(ids, minlength=self._num_envs).max() > 1:
            raise ValueError(
                f"ManagerBased reset-state {capability} env_ids contain duplicates: {ids.tolist()}"
            )
        return ids

    def _validate_columns(
        self,
        values: np.ndarray,
        *,
        width: int,
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray, "
                f"got {type(values).__name__}"
            )
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be a 1-D integer array"
            )
        columns = np.asarray(values, dtype=np.intp)
        if np.any(columns < 0) or np.any(columns >= width):
            raise IndexError(
                f"EventManager term '{term_name}' {capability} out of range for width "
                f"{width}: {columns.tolist()}"
            )
        if np.unique(columns).size != columns.size:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} contain duplicates: "
                f"{columns.tolist()}"
            )
        return columns

    def _validate_tensor_values(
        self,
        values: torch.Tensor,
        *,
        shape: tuple[int, ...],
        capability: str,
        term_name: str,
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if tuple(values.shape) != shape:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} has shape "
                f"{tuple(values.shape)}; expected {shape}"
            )
        if not values.is_floating_point():
            raise TypeError(f"EventManager term '{term_name}' {capability} must be floating")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"EventManager term '{term_name}' {capability} contains NaN or Inf")
        return values.contiguous()

    def _validate_values(
        self,
        values: np.ndarray | torch.Tensor,
        *,
        shape: tuple[int, ...],
        capability: str,
        term_name: str,
    ) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu":
                raise TypeError(
                    f"EventManager term '{term_name}' {capability} host write requires a "
                    f"CPU Torch tensor; got {values.device}"
                )
            if not values.is_contiguous():
                values = values.contiguous()
            values = values.numpy()
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be np.ndarray or a CPU "
                f"torch.Tensor, got {type(values).__name__}"
            )
        if values.shape != shape:
            raise ValueError(
                f"EventManager term '{term_name}' {capability} has shape {values.shape}; "
                f"expected {shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(
                f"EventManager term '{term_name}' {capability} must be floating, got {values.dtype}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"EventManager term '{term_name}' {capability} contains NaN or Inf")
        return values

    def _capability_error(
        self,
        term_name: str,
        capability: str,
        exc: BaseException,
    ) -> NotImplementedError:
        return NotImplementedError(
            f"EventManager term '{term_name}' reset-state capability '{capability}' is "
            f"unavailable on backend '{self._backend.backend_type}': {exc}"
        )

    def _require_active(self) -> None:
        if not (self._active or self._tensor_active):
            raise RuntimeError("ManagerBased reset-state mutation requires an active reset event")

    def _finish(self) -> None:
        self._active = False
        self._active_mask.fill(False)
        self._dirty_mask.fill(False)
        self._gain_dirty_mask.fill(False)
        for mask in self._randomization_dirty_masks.values():
            mask.fill(False)
        self._startup_randomization_fields.clear()
        for mask in self._mocap_masks.values():
            mask.fill(False)
        self._requesting_terms.clear()
        self._tensor_active = False
        self._tensor_active_mask = None
        self._mask_full_refresh = False
        self._tensor_has_writes = False
        self._tensor_owner_subset = False


__all__ = ["ResetStateTransaction"]
