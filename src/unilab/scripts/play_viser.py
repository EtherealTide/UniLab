"""Viser-based web viewer for trained policies.

Serves the playback rollout through a browser-based 3D viewer (powered by
*viser*) so no local display or GLFW is required. ``--sim`` selects the
physics owner; any backend declaring the physics-state playback contract
(mujoco, mjwarp, newton, drake, superdex) runs the rollout while per-env
MuJoCo playback models drive the rendered scene.

Usage:
    uv run python -m unilab.scripts.play_viser --algo ppo --task go2_joystick_flat --sim mujoco
    uv run python -m unilab.scripts.play_viser --algo appo --task go2_joystick_flat --sim mujoco \
      viser.port=8080 viser.max_envs=4

Camera controls (browser):
    Left-drag    - rotate
    Scroll       - zoom
    Right-drag   - pan
"""

import sys
from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from unilab.training import ensure_registries
from unilab.visualization.interactive_playback import PlayInteractiveArgs
from unilab.visualization.viser_playback import run_viser_playback_from_cfg

ensure_registries()

from unilab.scripts.play_interactive import (
    _build_play_args,
    _compose_interactive_config,
    _parse_interactive_cli,
    create_playback_session,
)


def play_viser(args: PlayInteractiveArgs, cfg: DictConfig, *, algo: str = "ppo") -> None:
    def log(message: str) -> None:
        print(f"[play_viser] {message}", flush=True)

    num_envs = int(OmegaConf.select(cfg, "viser.max_envs", default=16) or 16)
    session = create_playback_session(args, cfg, algo=algo, num_envs=num_envs, log=log)
    if session is None:
        return
    run_viser_playback_from_cfg(
        session[0],
        cfg,
        entrypoint="play_viser",
        start_paused=bool(getattr(args, "start_paused", False)),
        speed=float(getattr(args, "speed", 1.0)),
        log=log,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parsed = _parse_interactive_cli(sys.argv[1:] if argv is None else argv)
    cfg = _compose_interactive_config(parsed.algo, parsed.overrides)
    play_viser(_build_play_args(cfg, algo=parsed.algo), cfg, algo=parsed.algo)


if __name__ == "__main__":
    main()
