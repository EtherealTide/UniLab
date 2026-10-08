"""Fail-closed CUDA process-sharing probe tests."""

from __future__ import annotations

import platform
import socket
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unilab.training.cuda_process_sharing import probe_cuda_process_sharing


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _torch(uuid: str) -> Any:
    properties = SimpleNamespace(uuid=uuid)
    return SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda _index: properties,
        ),
    )


def _fake_commands(gpu_uuid: str = "GPU-a", *, server_pid: int = 2768293) -> Any:
    def run_command(command: list[str], **kwargs: Any) -> _Completed:
        del kwargs
        if command[0] == "nvidia-smi":
            return _Completed(f"0, {gpu_uuid}\n")
        if command == ["nvidia-cuda-mps-control", "get_server_list"]:
            return _Completed(f"{server_pid}\n")
        raise AssertionError(f"unexpected command: {command}")

    return run_command


@pytest.fixture
def linux_mps(monkeypatch: pytest.MonkeyPatch, short_unix_socket_root: Path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(short_unix_socket_root / "pipe"))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(short_unix_socket_root / "log"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a")
    control = short_unix_socket_root / "pipe" / "control"
    control.parent.mkdir(parents=True)
    control.touch()
    control.chmod(0o666)
    return control


def _socket_control(path: Any) -> None:
    path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    path.chmod(0o666)


def test_disabled_default_does_not_probe_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("disabled mode must not probe CUDA or MPS")

    monkeypatch.setattr(platform, "system", lambda: "not-linux")
    evidence = probe_cuda_process_sharing(
        None,
        "cpu",
        None,
        backend="mujoco",
        torch_module=None,
        run_command=reject,
    )

    assert evidence.manifest() == {
        "configured": None,
        "effective": None,
        "validated": False,
        "learner_device": "cpu",
        "collector_device": None,
        "learner_gpu_uuid": None,
        "collector_gpu_uuid": None,
        "control_pipe": None,
        "server_pid": None,
        "rank": 0,
        "world_size": 1,
    }


def test_valid_single_rank_request_records_server_evidence(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.manifest() == {
        "configured": "mps",
        "effective": "mps",
        "validated": True,
        "learner_device": "cuda:0",
        "collector_device": "cuda:0",
        "learner_gpu_uuid": "A",
        "collector_gpu_uuid": "A",
        "control_pipe": str(linux_mps),
        "server_pid": 2768293,
        "rank": 0,
        "world_size": 1,
    }


def test_valid_identity_compares_physical_uuid_not_ordinal(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.validated
    assert evidence.learner_gpu_uuid == evidence.collector_gpu_uuid


def test_single_host_dp_rank_records_rank_scoped_evidence(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    evidence = probe_cuda_process_sharing(
        "mps",
        "cuda:0",
        "cuda:0",
        backend="mjwarp",
        rank=1,
        world_size=2,
        torch_module=_torch("GPU-a"),
        run_command=_fake_commands(),
    )

    assert evidence.rank == 1
    assert evidence.world_size == 2
    assert evidence.validated


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested": "invalid"}, "expected null or 'mps'"),
        ({"backend": "mujoco"}, "only the mjwarp backend"),
        ({"rank": 2, "world_size": 2}, "valid rank in"),
        ({"collector_device": None}, "requires a CUDA collector"),
        ({"learner_device": "cpu"}, "requires CUDA learner and collector"),
    ],
)
def test_first_prerequisites_fail_closed(
    linux_mps: Any,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    del monkeypatch
    kwargs: dict[str, Any] = {
        "requested": "mps",
        "learner_device": "cuda:0",
        "collector_device": "cuda:0",
        "backend": "mjwarp",
        "world_size": 1,
        "torch_module": _torch("GPU-a"),
        "run_command": _fake_commands(),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        probe_cuda_process_sharing(**kwargs)


def test_non_linux_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    with pytest.raises(ValueError, match="requires Linux"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_unavailable_cuda_rejects(linux_mps: Any) -> None:
    unavailable = _torch("GPU-a")
    unavailable.cuda.is_available = lambda: False

    with pytest.raises(ValueError, match="CUDA is unavailable"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=unavailable,
            run_command=_fake_commands(),
        )


def test_gpu_identity_mismatch_rejects(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    with pytest.raises(ValueError, match=r"learner='A'.*collector='B'"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(gpu_uuid="GPU-b"),
        )


def test_missing_control_pipe_rejects_with_start_command(
    linux_mps: Any,
) -> None:
    linux_mps.unlink()

    with pytest.raises(ValueError, match="nvidia-cuda-mps-control -d"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_regular_file_is_not_a_control_pipe(linux_mps: Any) -> None:
    assert not stat.S_ISSOCK(linux_mps.stat().st_mode)

    with pytest.raises(ValueError, match="not a daemon control pipe"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=_fake_commands(),
        )


def test_unreachable_daemon_rejects_without_fallback(linux_mps: Any) -> None:
    _socket_control(linux_mps)

    def failing_daemon(command: list[str], **kwargs: Any) -> _Completed:
        del kwargs
        if command[0] == "nvidia-smi":
            return _Completed("0, GPU-a\n")
        return _Completed("Cannot find MPS control daemon process", returncode=1)

    with pytest.raises(ValueError, match="could not reach the control daemon"):
        probe_cuda_process_sharing(
            "mps",
            "cuda:0",
            "cuda:0",
            backend="mjwarp",
            torch_module=_torch("GPU-a"),
            run_command=failing_daemon,
        )
