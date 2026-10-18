"""Per-rank collector CPU affinity partitioning.

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15), Apache-2.0; previously
``uni_rl.ipc.dp_launcher``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

PhysicalCpuGroupDiscovery = Callable[[Sequence[int]], list[list[int]]]


def resolve_collector_cpu_ids(
    world_size: int,
    rank: int,
    cpu_count: int | None = None,
    explicit: Any = None,
    physical_cpu_group_discovery: PhysicalCpuGroupDiscovery | None = None,
) -> list[int] | None:
    """Resolve the CPU ids exclusively owned by this rank's collector.

    Returns None for the single-rank default (``world_size <= 1``) so the
    single-card path stays bit-identical to the pre-partition behavior.
    With ``cpu_count=None`` (the normal upper-layer path), it discovers the
    current affinity and assigns complete physical-core groups, including SMT
    siblings, to each rank. ``cpu_count`` retains the deterministic logical
    CPU fallback used by compatibility callers and tests. ``explicit``
    (``training.dp_collector_cpu_ids``, one segment per rank) overrides either
    automatic route; segments must number exactly ``world_size``, be non-empty,
    contain non-negative ints, and not overlap. An empty ``explicit`` falls
    back to automatic partitioning.

    Cold path only: call at runner construction, never from the collect loop.
    """
    world_size = int(world_size)
    if world_size <= 1:
        return None
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"data-parallel rank {rank} is out of range for world_size={world_size}")
    groups: list[list[int]] | None = None
    if cpu_count is None:
        if physical_cpu_group_discovery is None:
            physical_cpu_group_discovery = discover_physical_cpu_groups
        sched_getaffinity = getattr(os, "sched_getaffinity", None)
        if sched_getaffinity is not None:
            try:
                available_ids = sorted(sched_getaffinity(0))
            except OSError:
                available_ids = list(range(os.cpu_count() or 1))
        else:
            available_ids = list(range(os.cpu_count() or 1))
        groups = physical_cpu_group_discovery(available_ids)
        cpu_count = len(available_ids)
    cpu_count = int(cpu_count or 0)
    if cpu_count < world_size:
        raise ValueError(
            f"cpu_count={cpu_count} cannot give each data-parallel rank its own CPU "
            f"(world_size={world_size})"
        )
    if explicit is not None and len(explicit) > 0:
        segments: list[list[int]] = [list(segment) for segment in explicit]
        if len(segments) != world_size:
            raise ValueError(
                f"training.dp_collector_cpu_ids must provide exactly world_size={world_size} "
                f"segments, got {len(segments)}"
            )
        seen: set[int] = set()
        for segment in segments:
            if not segment:
                raise ValueError(
                    f"training.dp_collector_cpu_ids segments must be non-empty, got {segments!r}"
                )
            for cpu_id in segment:
                if isinstance(cpu_id, bool) or not isinstance(cpu_id, int) or cpu_id < 0:
                    raise ValueError(
                        "training.dp_collector_cpu_ids entries must be non-negative "
                        f"integers, got {cpu_id!r}"
                    )
                if cpu_id in seen:
                    raise ValueError(
                        f"training.dp_collector_cpu_ids segments overlap on CPU {cpu_id}"
                    )
                seen.add(cpu_id)
        return segments[rank]
    if groups:
        if len(groups) < world_size:
            raise ValueError(
                f"available physical CPU groups={len(groups)} cannot serve world_size={world_size}"
            )
        group_count = len(groups)
        start = rank * group_count // world_size
        end = (rank + 1) * group_count // world_size
        if end <= start:
            raise ValueError(
                f"available physical CPU groups={group_count} cannot serve world_size={world_size}"
            )
        owned = groups[start:end]
        return [cpu_id for group in owned for cpu_id in group]
    size = cpu_count // world_size
    return list(range(rank * size, rank * size + size))


def discover_physical_cpu_groups(cpu_ids: Sequence[int]) -> list[list[int]]:
    """Discover physical-core groups for topology-aware CPU partitioning.

    This is the public test/assembly seam for the host-topology reader. Prefer
    injecting a deterministic provider into ``resolve_collector_cpu_ids``; patch
    this function only for callers that cannot thread the parameter through.
    """
    ids = list(cpu_ids)
    if os.name == "posix" and os.path.isdir("/sys/devices/system/cpu"):
        groups: dict[tuple[str, str], list[int]] = {}
        try:
            for cpu_id in ids:
                with open(
                    f"/sys/devices/system/cpu/cpu{cpu_id}/topology/physical_package_id"
                ) as package_file:
                    package = package_file.read().strip()
                with open(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id") as core_file:
                    core = core_file.read().strip()
                groups.setdefault((package, core), []).append(cpu_id)
            if groups:
                return list(groups.values())
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            physical = int(
                subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
            )
            physical = max(1, min(physical, len(ids)))
            darwin_groups: list[list[int]] = [[] for _ in range(physical)]
            for index, cpu_id in enumerate(ids):
                darwin_groups[min(index * physical // len(ids), physical - 1)].append(cpu_id)
            return [group for group in darwin_groups if group]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return [[cpu_id] for cpu_id in ids]


__all__ = [
    "PhysicalCpuGroupDiscovery",
    "discover_physical_cpu_groups",
    "resolve_collector_cpu_ids",
]
