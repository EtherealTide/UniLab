"""Shared-memory tensor ring for APPO rollouts.

The storage is a persistent, tensor-owned host bridge.  CPU-authoritative
collectors write directly into these tensors; device-resident collectors make
one explicit D2H copy into a ring slot.  Learners copy slots into their own
device-resident staging pool.  Every cross-device transfer is therefore an
explicit ``copy_`` at this boundary rather than a hidden per-step NumPy
conversion.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Mapping

import torch

_SPAWN_CTX = mp.get_context("spawn")

_FIELD_SHAPES = {
    "obs": lambda ne, ns, od, ad, cd: (ne, ns, od),
    "critic": lambda ne, ns, od, ad, cd: (ne, ns, cd),
    "actions": lambda ne, ns, od, ad, cd: (ne, ns, ad),
    "log_probs": lambda ne, ns, od, ad, cd: (ne, ns),
    "rewards": lambda ne, ns, od, ad, cd: (ne, ns),
    "dones": lambda ne, ns, od, ad, cd: (ne, ns),
    "truncated": lambda ne, ns, od, ad, cd: (ne, ns),
    "last_obs": lambda ne, ns, od, ad, cd: (ne, od),
    "last_critic": lambda ne, ns, od, ad, cd: (ne, cd),
}


class TensorRolloutRingBuffer:
    """Bounded multi-process rollout ring backed by shared Torch tensors."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        critic_dim: int = 0,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Mapping[str, str] | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.critic_dim = int(critic_dim)
        self.num_slots = int(num_slots)
        if min(self.num_envs, self.num_steps, self.obs_dim, self.action_dim, self.num_slots) < 1:
            raise ValueError("TensorRolloutRingBuffer dimensions must be positive")
        if self.critic_dim < 0:
            raise ValueError("critic_dim must be non-negative")

        self._shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        fields = dict(_FIELD_SHAPES)
        if self.critic_dim == 0:
            fields.pop("critic")
            fields.pop("last_critic")

        for field, shape_fn in fields.items():
            slot_shape = shape_fn(
                self.num_envs, self.num_steps, self.obs_dim, self.action_dim, self.critic_dim
            )
            shape = (self.num_slots, *slot_shape)
            count = 1
            for extent in shape:
                count *= int(extent)
            size = max(count * torch.empty((), dtype=torch.float32).element_size(), 1)
            if create:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                if shm_name_prefix is None or field not in shm_name_prefix:
                    raise ValueError(f"missing shared-memory name for rollout field {field!r}")
                shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)
            self._shm_blocks[field] = shm
            # Some platforms round shared-memory allocations up to page granularity.
            # Bind only the logical payload; padding must not affect the shape.
            self._tensors[field] = torch.frombuffer(
                shm.buf,
                dtype=torch.float32,
                count=count,
            ).reshape(shape)

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    @property
    def name(self) -> dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def slot_shapes(self) -> dict[str, tuple[int, ...]]:
        return {field: tuple(tensor.shape[1:]) for field, tensor in self._tensors.items()}

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    def _clamp_read_ptr_to_valid_window(self) -> None:
        write = int(self._write_ptr.value)
        oldest = max(0, write - self.num_slots)
        if int(self._read_ptr.value) >= oldest:
            return
        with self._read_ptr.get_lock():
            if int(self._read_ptr.value) < oldest:
                self._read_ptr.value = oldest

    @property
    def write_slot(self) -> int:
        return int(self._write_ptr.value) % self.num_slots

    @property
    def write_buffer(self) -> dict[str, torch.Tensor]:
        slot = self.write_slot
        return {field: tensor[slot] for field, tensor in self._tensors.items()}

    def signal_write_done(self) -> None:
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        self._clamp_read_ptr_to_valid_window()
        return min(
            max(0, int(self._write_ptr.value) - int(self._read_ptr.value)),
            self.num_slots,
        )

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        self._clamp_read_ptr_to_valid_window()
        return int(self._read_ptr.value) % self.num_slots

    def read_tensor_views(self) -> dict[str, torch.Tensor]:
        """Return borrowed tensor views for the current read slot."""
        slot = self.read_slot
        return {field: tensor[slot] for field, tensor in self._tensors.items()}

    def advance_read(self) -> None:
        with self._read_ptr.get_lock():
            write = int(self._write_ptr.value)
            next_read = min(int(self._read_ptr.value) + 1, write)
            oldest = max(0, write - self.num_slots)
            self._read_ptr.value = max(next_read, oldest)

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass


__all__ = ["TensorRolloutRingBuffer"]
