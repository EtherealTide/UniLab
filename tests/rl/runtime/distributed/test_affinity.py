"""Per-rank collector CPU affinity tests (owner: rl.runtime.distributed.affinity).

Merged from UniLab's former ``tests/ipc/test_dp_launcher.py``.
"""

from __future__ import annotations

import pytest

import unilab.rl.runtime.distributed.affinity as affinity
from unilab.rl.runtime.distributed.affinity import (
    discover_physical_cpu_groups,
    resolve_collector_cpu_ids,
)


def test_physical_cpu_topology_discovery_is_a_public_seam() -> None:
    assert callable(discover_physical_cpu_groups)


def test_resolve_collector_cpu_ids_uses_public_topology_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setattr(affinity, "discover_physical_cpu_groups", lambda _: groups)

    assert resolve_collector_cpu_ids(2, 0) == [0, 4, 1, 5]
    assert resolve_collector_cpu_ids(2, 1) == [2, 6, 3, 7]


def test_resolve_collector_cpu_ids_single_rank_returns_none():
    assert resolve_collector_cpu_ids(1, 0, 128) is None
    assert resolve_collector_cpu_ids(0, 0, 128) is None


def test_resolve_collector_cpu_ids_even_partition():
    assert resolve_collector_cpu_ids(2, 0, 128) == list(range(0, 64))
    assert resolve_collector_cpu_ids(2, 1, 128) == list(range(64, 128))


def test_resolve_collector_cpu_ids_keeps_physical_siblings_together(
    monkeypatch: pytest.MonkeyPatch,
):
    groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert resolve_collector_cpu_ids(2, 0, physical_cpu_group_discovery=lambda _: groups) == [
        0,
        4,
        1,
        5,
    ]
    assert resolve_collector_cpu_ids(2, 1, physical_cpu_group_discovery=lambda _: groups) == [
        2,
        6,
        3,
        7,
    ]


def test_resolve_collector_cpu_ids_remainder_stays_unassigned():
    # 129 CPUs / 2 ranks -> 64+64; CPU 128 keeps default OS scheduling.
    assert resolve_collector_cpu_ids(2, 0, 129) == list(range(0, 64))
    assert resolve_collector_cpu_ids(2, 1, 129) == list(range(64, 128))


def test_resolve_collector_cpu_ids_requires_one_cpu_per_rank():
    with pytest.raises(ValueError, match="cpu_count"):
        resolve_collector_cpu_ids(4, 0, 3)


def test_resolve_collector_cpu_ids_rank_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_collector_cpu_ids(2, 2, 128)


def test_resolve_collector_cpu_ids_explicit_valid():
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=[[0, 1], [2, 3]]) == [2, 3]


def test_resolve_collector_cpu_ids_empty_explicit_falls_back_to_auto():
    assert resolve_collector_cpu_ids(2, 1, 128, explicit=[]) == list(range(64, 128))


def test_resolve_collector_cpu_ids_explicit_rejects_segment_count_mismatch():
    with pytest.raises(ValueError, match="world_size"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0], [1], [2]])


def test_resolve_collector_cpu_ids_explicit_rejects_overlapping_segments():
    with pytest.raises(ValueError, match="overlap"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, 1], [1, 2]])


def test_resolve_collector_cpu_ids_explicit_rejects_empty_segment():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, 1], []])


@pytest.mark.parametrize("bad_id", [-1, True, "2"])
def test_resolve_collector_cpu_ids_explicit_rejects_invalid_ids(bad_id):
    with pytest.raises(ValueError, match="non-negative integers"):
        resolve_collector_cpu_ids(2, 0, 128, explicit=[[0, bad_id], [2, 3]])
