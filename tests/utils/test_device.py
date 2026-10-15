"""Host hardware-info collection tests (owner: unilab.utils.device).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15) ``tests/utils/test_device.py``;
the implementation stayed in UniLab as ``unilab.utils.device``.
"""

from __future__ import annotations

import unilab.utils.device as device_mod


def test_linux_device_info_reads_amd_visible_vram(monkeypatch) -> None:
    cpuinfo = "model name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\nprocessor\t: 0\n"
    meminfo = "MemTotal:       32486180 kB\n"
    amd_smi_metric = """
GPU: 0
    MEM_USAGE:
        TOTAL_VRAM: 98304 MB
        USED_VRAM: 3603 MB
        FREE_VRAM: 94701 MB
        TOTAL_VISIBLE_VRAM: 98304 MB
        USED_VISIBLE_VRAM: 3603 MB
        FREE_VISIBLE_VRAM: 94701 MB
        TOTAL_GTT: 15862 MB
        USED_GTT: 158 MB
        FREE_GTT: 15704 MB
"""

    def fake_open(path, *args, **kwargs):
        del args, kwargs
        if path == "/proc/cpuinfo":
            from io import StringIO

            return StringIO(cpuinfo)
        if path == "/proc/meminfo":
            from io import StringIO

            return StringIO(meminfo)
        raise FileNotFoundError(path)

    def fake_check_output(cmd, *args, **kwargs):
        del args, kwargs
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError(cmd[0])
        if cmd[:2] == ["rocm-smi", "--showproductname"]:
            return "Card series: AMD Radeon Graphics\n"
        if cmd[:2] == ["amd-smi", "metric"]:
            return amd_smi_metric
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(device_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(device_mod.subprocess, "check_output", fake_check_output)

    info = device_mod._get_device_info_linux()

    assert info["gpu_name"] == "Radeon 8060S"
    assert info["gpu_memory"] == "98304 MB"
    assert info["gpu_gtt_memory"] == "15862 MB"
    assert info["memory"] == "31.0 GB"


def test_linux_device_info_reads_intel_igpu_from_lspci(monkeypatch) -> None:
    cpuinfo = "model name\t: Intel(R) Core(TM) Ultra 9 185H\nprocessor\t: 0\n"
    meminfo = "MemTotal:       31692928 kB\n"
    lspci_out = (
        "00:02.0 VGA compatible controller: "
        "Intel Corporation Meteor Lake-P [Intel Arc Graphics] (rev 08)\n"
    )

    def fake_open(path, *args, **kwargs):
        del args, kwargs
        from io import StringIO

        if path == "/proc/cpuinfo":
            return StringIO(cpuinfo)
        if path == "/proc/meminfo":
            return StringIO(meminfo)
        raise FileNotFoundError(path)

    def fake_check_output(cmd, *args, **kwargs):
        del args, kwargs
        if cmd[:1] == ["lspci"]:
            return lspci_out
        # No nvidia-smi, rocm-smi, or amd-smi on Intel iGPU systems
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(device_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(device_mod.subprocess, "check_output", fake_check_output)

    info = device_mod._get_device_info_linux()

    assert info["gpu_name"] == "Intel Arc Graphics"
    assert info["chip"] == "Intel(R) Core(TM) Ultra 9 185H"
    assert info["memory"] == "30.2 GB"


def test_get_device_info_dict_reports_core_fields() -> None:
    info = device_mod.get_device_info_dict()
    assert info["platform"]
    assert "chip" in info
    assert "cpu_total_cores" in info
    assert "memory" in info
    assert "gpu_name" in info or "gpu_cores" in info
