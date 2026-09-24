"""Viser-based interactive viewer for trained policies.

Thin wrapper around the packaged viewer ``unilab.scripts.play_viser``; the
playback loop, session dispatch, and scene code live there. This script keeps
the direct Hydra entry for source checkouts. viser is a required dependency.

Usage::

    # Zero-action mode (no checkpoint needed)
    uv run scripts/play_viser.py task=go2_joystick_flat/mujoco interactive.action_mode=zero

    # With a trained policy
    uv run scripts/play_viser.py task=go2_joystick_flat/mujoco interactive.action_mode=policy

    # Multiple environments with env switching
    uv run scripts/play_viser.py task=go2_joystick_flat/mujoco algo.num_envs=4 viser.port=8080

    # Motion tracking task
    uv run scripts/play_viser.py task=g1_motion_tracking/mujoco interactive.action_mode=policy

Camera controls (browser):
    Left-drag    - rotate
    Scroll       - zoom
    Right-drag   - pan
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import DictConfig

from unilab.scripts.play_interactive import _build_play_args
from unilab.scripts.play_viser import play_viser


@hydra.main(version_base="1.3", config_path="../src/unilab/conf/ppo", config_name="config")
def main(cfg: DictConfig) -> None:
    play_viser(_build_play_args(cfg, algo="ppo"), cfg, algo="ppo")


if __name__ == "__main__":
    main()
