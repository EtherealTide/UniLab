"""Pure inference-placement resolution (owner: rl.runtime.placement).

Merged from unilab_rl @ 3654e4b (unilab-rl 1.4.15)
``tests/utils/test_tensor_runtime.py``; Hydra-level transport resolution
coverage lives in ``tests/training/test_tensor_runtime.py``.
"""

from __future__ import annotations

import pytest

from unilab.rl.runtime.placement import resolve_inference_placement


def test_inference_transport_request_must_be_named_device() -> None:
    with pytest.raises(ValueError, match="inference transport must be one of"):
        resolve_inference_placement(
            learner_device="cuda:0",
            transport="accel",
            env_tensor_native=True,
            algo_name="SAC",
        )
