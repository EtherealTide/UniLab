"""Tests for the tensor-native APPO rollout IPC primitive."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory

import torch

from unilab.rl.runtime.ipc import TensorRolloutRingBuffer

_NUM_ENVS = 4
_NUM_STEPS = 2
_OBS_DIM = 3
_ACTION_DIM = 2
_CRITIC_DIM = 5
_NUM_SLOTS = 2

_EXPECTED_FIELDS = {
    "obs",
    "critic",
    "actions",
    "log_probs",
    "rewards",
    "dones",
    "truncated",
    "last_obs",
    "last_critic",
}

_SPAWN_CTX = mp.get_context("spawn")


def _make_ring(num_slots: int = _NUM_SLOTS) -> TensorRolloutRingBuffer:
    return TensorRolloutRingBuffer(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        num_slots=num_slots,
        create=True,
    )


def _attach_to_padded_blocks() -> TensorRolloutRingBuffer:
    """Attach to allocations containing more than one extra padding float."""
    slot_shapes = {
        "obs": (_NUM_ENVS, _NUM_STEPS, _OBS_DIM),
        "critic": (_NUM_ENVS, _NUM_STEPS, _CRITIC_DIM),
        "actions": (_NUM_ENVS, _NUM_STEPS, _ACTION_DIM),
        "log_probs": (_NUM_ENVS, _NUM_STEPS),
        "rewards": (_NUM_ENVS, _NUM_STEPS),
        "dones": (_NUM_ENVS, _NUM_STEPS),
        "truncated": (_NUM_ENVS, _NUM_STEPS),
        "last_obs": (_NUM_ENVS, _OBS_DIM),
        "last_critic": (_NUM_ENVS, _CRITIC_DIM),
    }
    blocks = {
        field: shared_memory.SharedMemory(
            create=True,
            size=(int(torch.tensor(shape).prod().item()) + 2) * 4 + 1,
        )
        for field, shape in slot_shapes.items()
    }
    sync_primitives = (_SPAWN_CTX.Value("l", 0), _SPAWN_CTX.Value("l", 0))
    ring = TensorRolloutRingBuffer(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        num_slots=1,
        create=False,
        shm_name_prefix={field: block.name for field, block in blocks.items()},
    )
    ring.attach_sync_primitives(*sync_primitives)
    return ring


def test_slot_shapes_and_tensor_carriers() -> None:
    ring = _make_ring()

    assert ring.slot_shapes == {
        "obs": (_NUM_ENVS, _NUM_STEPS, _OBS_DIM),
        "critic": (_NUM_ENVS, _NUM_STEPS, _CRITIC_DIM),
        "actions": (_NUM_ENVS, _NUM_STEPS, _ACTION_DIM),
        "log_probs": (_NUM_ENVS, _NUM_STEPS),
        "rewards": (_NUM_ENVS, _NUM_STEPS),
        "dones": (_NUM_ENVS, _NUM_STEPS),
        "truncated": (_NUM_ENVS, _NUM_STEPS),
        "last_obs": (_NUM_ENVS, _OBS_DIM),
        "last_critic": (_NUM_ENVS, _CRITIC_DIM),
    }
    write = ring.write_buffer
    assert set(write) == _EXPECTED_FIELDS
    assert all(isinstance(value, torch.Tensor) for value in write.values())
    ring.cleanup()


def test_write_read_and_overwrite_window() -> None:
    ring = _make_ring(num_slots=2)

    for value in (1.0, 2.0, 3.0):
        write = ring.write_buffer
        for tensor in write.values():
            tensor.fill_(value)
        ring.signal_write_done()

    assert ring.available() == 2
    first = ring.read_tensor_views()
    assert torch.all(first["obs"] == 2.0)
    ring.advance_read()

    second = ring.read_tensor_views()
    assert torch.all(second["actions"] == 3.0)
    assert torch.all(second["last_critic"] == 3.0)
    ring.cleanup()


def test_attach_reads_owner_data_and_preserves_views() -> None:
    owner = _make_ring()
    write = owner.write_buffer
    for tensor in write.values():
        tensor.fill_(7.0)
    owner.signal_write_done()

    attached = TensorRolloutRingBuffer(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=_CRITIC_DIM,
        num_slots=_NUM_SLOTS,
        create=False,
        shm_name_prefix=owner.name,
    )
    attached.attach_sync_primitives(owner._write_ptr, owner._read_ptr)
    read = attached.read_tensor_views()
    assert torch.all(read["obs"] == 7.0)
    assert all(isinstance(value, torch.Tensor) for value in read.values())

    attached.close()
    owner.cleanup()


def test_padded_allocations_bind_only_to_payload_shapes() -> None:
    ring = _attach_to_padded_blocks()

    assert ring.slot_shapes == {
        "obs": (_NUM_ENVS, _NUM_STEPS, _OBS_DIM),
        "critic": (_NUM_ENVS, _NUM_STEPS, _CRITIC_DIM),
        "actions": (_NUM_ENVS, _NUM_STEPS, _ACTION_DIM),
        "log_probs": (_NUM_ENVS, _NUM_STEPS),
        "rewards": (_NUM_ENVS, _NUM_STEPS),
        "dones": (_NUM_ENVS, _NUM_STEPS),
        "truncated": (_NUM_ENVS, _NUM_STEPS),
        "last_obs": (_NUM_ENVS, _OBS_DIM),
        "last_critic": (_NUM_ENVS, _CRITIC_DIM),
    }
    write = ring.write_buffer
    assert set(write) == _EXPECTED_FIELDS
    for tensor in write.values():
        tensor.fill_(5.0)
    ring.signal_write_done()
    assert all(torch.all(value == 5.0) for value in ring.read_tensor_views().values())

    for shm in ring._shm_blocks.values():
        shm.unlink()
    ring.close()


def test_padded_optional_critic_fields_are_omitted() -> None:
    dimensions = {
        "obs": (_NUM_STEPS, _OBS_DIM),
        "actions": (_NUM_STEPS, _ACTION_DIM),
        "log_probs": (_NUM_STEPS,),
        "rewards": (_NUM_STEPS,),
        "dones": (_NUM_STEPS,),
        "truncated": (_NUM_STEPS,),
        "last_obs": (_OBS_DIM,),
    }
    blocks = {
        field: shared_memory.SharedMemory(
            create=True,
            size=(_NUM_ENVS * int(torch.tensor(shape).prod().item()) + 2) * 4 + 1,
        )
        for field, shape in dimensions.items()
    }
    ring = TensorRolloutRingBuffer(
        num_envs=_NUM_ENVS,
        num_steps=_NUM_STEPS,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        critic_dim=0,
        num_slots=1,
        create=False,
        shm_name_prefix={field: block.name for field, block in blocks.items()},
    )

    assert "critic" not in ring.slot_shapes
    assert "last_critic" not in ring.slot_shapes

    for shm in ring._shm_blocks.values():
        shm.unlink()
    ring.close()


def test_wait_for_data_times_out_when_empty() -> None:
    ring = _make_ring()
    assert ring.wait_for_data(timeout=0.01) is False
    ring.cleanup()


def test_invalid_dimensions_fail_closed() -> None:
    for kwargs in ({"num_envs": 0}, {"num_steps": -1}, {"obs_dim": 0}, {"critic_dim": -1}):
        try:
            TensorRolloutRingBuffer(
                num_envs=kwargs.get("num_envs", _NUM_ENVS),
                num_steps=kwargs.get("num_steps", _NUM_STEPS),
                obs_dim=kwargs.get("obs_dim", _OBS_DIM),
                action_dim=_ACTION_DIM,
                critic_dim=kwargs.get("critic_dim", 0),
                num_slots=1,
                create=True,
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
