from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.tools.sync_tensor_workspace import load_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tensor_runtime_workspace.json"
SCRIPT = ROOT / "scripts/tools/sync_tensor_workspace.py"


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_tensor_workspace_manifest_is_pinned() -> None:
    siblings = load_manifest(MANIFEST, ROOT)

    assert [sibling.name for sibling in siblings] == [
        "unisim-core",
        "unilab-rl",
        "mjbatch-uni",
    ]
    assert all(len(sibling.commit) == 40 for sibling in siblings)
    assert all(sibling.ref == "develop/tensor-runtime" for sibling in siblings)
    unisim = siblings[0]
    assert unisim.sentinel_env == "UNILAB_LOCAL_UNISIM"
    assert unisim.path == (ROOT / "../unisim").resolve()


def test_tensor_workspace_sync_clones_pinned_commit_and_rejects_dirty_siblings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(["init"], source)
    _git(["checkout", "-b", "main"], source)
    _git(["config", "user.email", "test@example.com"], source)
    _git(["config", "user.name", "Tensor Workspace Test"], source)
    (source / "pyproject.toml").write_text("[project]\nname = 'sibling'\n", encoding="utf-8")
    _git(["add", "."], source)
    _git(["commit", "-m", "initial"], source)
    commit = _git(["rev-parse", "HEAD"], source)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "tensor_runtime_workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "local-editable",
                "integration_branch": "main",
                "siblings": [
                    {
                        "name": "sibling",
                        "repository": str(source),
                        "ref": "main",
                        "commit": commit,
                        "path": "sibling",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--workspace-root",
            str(workspace),
            "--sync",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checkout = workspace / "sibling"
    assert f"sibling: {commit} (pinned)" in result.stdout
    assert _git(["rev-parse", "HEAD"], checkout) == commit

    (checkout / "local-change").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as error:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest),
                "--workspace-root",
                str(workspace),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "sibling checkout at" in error.value.stderr or "is dirty" in error.value.stderr
