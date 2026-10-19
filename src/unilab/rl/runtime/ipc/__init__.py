"""IPC primitives for multi-process RL training."""

from typing import TYPE_CHECKING, Any

from unilab.rl.runtime.ipc.replay_buffer import ReplayBuffer
from unilab.rl.runtime.ipc.tensor_rollout_ring_buffer import TensorRolloutRingBuffer
from unilab.rl.runtime.ipc.weight_sync import SharedWeightSync

if TYPE_CHECKING:
    from unilab.rl.runtime.async_runner import AsyncRunner

__all__ = [
    "SharedWeightSync",
    "TensorRolloutRingBuffer",
    "AsyncRunner",
    "ReplayBuffer",
]


def __getattr__(name: str) -> Any:
    # AsyncRunner lives one level up (unilab.rl.runtime.async_runner) and
    # imports back into this package; resolve it lazily so either import
    # order works.
    if name == "AsyncRunner":
        from unilab.rl.runtime.async_runner import AsyncRunner

        return AsyncRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
