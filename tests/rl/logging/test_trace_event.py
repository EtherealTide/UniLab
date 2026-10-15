from __future__ import annotations

import json

import torch

from unilab.rl.logging.trace_event import TraceRecorder


def test_trace_json_renders_torch_diagnostic_values(tmp_path):
    recorder = TraceRecorder("trace-test")
    recorder.add_slice(
        "diagnostic",
        category="trace",
        start_ns=1,
        end_ns=2,
        args={
            "device": torch.device("cuda:0"),
            "dtype": torch.float32,
            "tensor": torch.zeros((2, 3)),
        },
    )

    path = recorder.write_json(tmp_path / "trace.json")
    payload = json.loads(path.read_text())
    event = next(event for event in payload["traceEvents"] if event["name"] == "diagnostic")

    assert event["args"]["device"] == "cuda:0"
    assert event["args"]["dtype"] == "torch.float32"
    assert event["args"]["tensor"] == "torch.float32 (2, 3) on cpu"
