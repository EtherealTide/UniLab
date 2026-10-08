"""Newton optional-runtime identity and routing boundaries."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def test_newton_import_path_does_not_eagerly_import_engine_modules() -> None:
    pytest.importorskip("unisim.backend.newton")
    code = textwrap.dedent(
        """
        import sys

        from unilab.base.backend_factory import create_backend
        from unisim.backend.newton import NewtonBackend

        assert create_backend is not None
        assert NewtonBackend is not None
        print("newton", "newton" in sys.modules)
        print("mujoco_warp", "mujoco_warp" in sys.modules)
        print("warp", "warp" in sys.modules)
        print("mujoco", "mujoco" in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "newton False",
        "mujoco_warp False",
        "warp False",
        "mujoco False",
    ]
