from __future__ import annotations

import pytest
from scripts.benchmark.core.task_names import (
    canonical_locomotion_task_ids,
    normalize_locomotion_task_id,
)


@pytest.mark.parametrize(
    "task_id",
    [
        "go2_joystick_flat",
        "g1_walk_flat",
        " go2_joystick_flat ",
        " g1_walk_flat ",
    ],
)
def test_normalize_accepts_canonical_owner_task_ids(task_id: str) -> None:
    assert normalize_locomotion_task_id(task_id) == task_id.strip()


@pytest.mark.parametrize(
    "removed_input",
    [
        "Go2JoystickFlat",
        "G1WalkFlat",
        "task=go2_joystick_flat/mujoco",
        "task=g1_walk_flat/mujoco",
        "go2_joystick_flat/mujoco",
        "g1_walk_flat/mujoco",
        "task=go2_joystick_flat",
        "go2",
    ],
)
def test_normalize_rejects_removed_compatibility_inputs(removed_input: str) -> None:
    with pytest.raises(ValueError, match="Available task ids"):
        normalize_locomotion_task_id(removed_input)


def test_normalize_error_lists_only_current_usage() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_locomotion_task_id("G1WalkFlat")

    message = str(exc_info.value)
    assert canonical_locomotion_task_ids() == ["go2_joystick_flat", "g1_walk_flat"]
    assert "legacy" not in message
    assert "aliases" not in message
    assert "Pass one of those owner task ids verbatim." in message
