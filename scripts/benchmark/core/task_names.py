from __future__ import annotations

from dataclasses import dataclass

from unilab.envs import ManagerBasedRlEnvCfg


@dataclass(frozen=True)
class LocomotionTaskSpec:
    owner_task_id: str
    display_name: str
    config_cls: type
    model_file: str | None = None


_TASK_SPECS = {
    "go2_joystick_flat": LocomotionTaskSpec(
        owner_task_id="go2_joystick_flat",
        display_name="go2_joystick_flat",
        config_cls=ManagerBasedRlEnvCfg,
        model_file="src/unilab/assets/robots/go2/scene_flat.xml",
    ),
    "g1_walk_flat": LocomotionTaskSpec(
        owner_task_id="g1_walk_flat",
        display_name="g1_walk_flat",
        config_cls=ManagerBasedRlEnvCfg,
        model_file="src/unilab/assets/robots/g1/scene_flat.xml",
    ),
}


def canonical_locomotion_task_ids() -> list[str]:
    return list(_TASK_SPECS.keys())


def normalize_locomotion_task_id(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.endswith("/motrix"):
        raise ValueError(
            f"Task '{task_name}' targets motrix, but this benchmark only measures MuJoCo paths."
        )
    if normalized in _TASK_SPECS:
        return normalized
    raise ValueError(
        f"Unknown task '{task_name}'. Available task ids: {canonical_locomotion_task_ids()}. "
        "Pass one of those owner task ids verbatim."
    )


def locomotion_task_spec(task_name: str) -> LocomotionTaskSpec:
    return _TASK_SPECS[normalize_locomotion_task_id(task_name)]


def locomotion_task_model_file(task_name: str) -> str:
    spec = locomotion_task_spec(task_name)
    if spec.model_file is not None:
        return spec.model_file
    cfg = spec.config_cls()
    scene = getattr(cfg, "scene", None)
    model_file = getattr(scene, "model_file", None)
    if model_file:
        return str(model_file)

    raise ValueError(f"{type(cfg).__name__} does not define scene.model_file")
