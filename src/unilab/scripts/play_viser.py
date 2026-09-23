"""Viser-based web viewer for trained policies.

Serves the playback rollout through a browser-based 3D viewer (powered by
*viser*) so no local display or GLFW is required. ``--sim`` selects the
physics owner; mjwarp runs the rollout while per-env MuJoCo playback models
drive the rendered scene.

Usage:
    uv run python -m unilab.scripts.play_viser --algo ppo --task go2_joystick_flat --sim mujoco
    uv run python -m unilab.scripts.play_viser --algo appo --task go2_joystick_flat --sim mujoco \
      viser.port=8080 viser.max_envs=4

Prerequisites: ``uv sync --extra viser``.

Camera controls (browser):
    Left-drag    - rotate
    Scroll       - zoom
    Right-drag   - pan
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.training import ensure_registries
from unilab.visualization.interactive_playback import PlaybackControls, PlayInteractiveArgs
from unilab.visualization.playback_state import (
    PhysicsStateApplier,
    assert_physics_state_playback_supported,
)
from unilab.visualization.render_many import get_grid_offsets
from unilab.visualization.viser_scene import (
    VISER_AVAILABLE,
    MujocoViserBatchScene,
    MujocoViserScene,
    build_visible_env_indices,
)

ensure_registries()

from unilab.scripts.play_interactive import (
    _build_play_args,
    _compose_interactive_config,
    _load_mujoco_model_file_for_viewer,
    _parse_interactive_cli,
    create_playback_session,
)

if VISER_AVAILABLE:
    import viser
else:
    viser = None

import mujoco


def _load_env_playback_model(env: Any, env_index: int):
    """Resolve the MuJoCo playback model for one env.

    Backends either return an ``MjModel`` directly (mujoco) or a model file
    path (mjwarp per-env variants); both are accepted.
    """
    model = env.get_playback_model(env_index)
    if isinstance(model, mujoco.MjModel):
        return model
    if isinstance(model, str):
        return _load_mujoco_model_file_for_viewer(model)
    raise TypeError(f"Expected mujoco.MjModel or model file path for playback, got {type(model)!r}")


def _scene_offset(offset_xy: np.ndarray) -> tuple[float, float, float]:
    """Convert a 2D grid offset into a 3D scene offset."""
    return (float(offset_xy[0]), float(offset_xy[1]), 0.0)


def _build_scene_entries(
    server: Any,
    env: Any,
    *,
    mode: str,
    selected_visible_idx: int,
    visible_env_indices: np.ndarray,
    spacing: float,
) -> list[dict[str, Any]]:
    """Construct viser scenes and MuJoCo data objects for active env views."""
    entries: list[dict[str, Any]] = []
    if mode == "single":
        env_idx = int(visible_env_indices[selected_visible_idx])
        mj_model = _load_env_playback_model(env, env_idx)
        entries.append(
            {
                "slot_idx": selected_visible_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "applier": PhysicsStateApplier(env, mj_model, env_index=env_idx),
                "scene": MujocoViserScene(server, mj_model, name_prefix="/mujoco/single"),
            }
        )
        return entries

    offsets = get_grid_offsets(len(visible_env_indices), spacing=spacing)
    models: list[mujoco.MjModel] = []
    data: list[mujoco.MjData] = []
    appliers: list[PhysicsStateApplier] = []
    for env_idx in visible_env_indices:
        model = _load_env_playback_model(env, int(env_idx))
        models.append(model)
        data.append(mujoco.MjData(model))
        appliers.append(PhysicsStateApplier(env, model, env_index=int(env_idx)))

    # MuJoCo task instances normally share one model.  Viser can then render
    # each geom as a batched mesh, reducing per-frame messages from
    # O(environments * geoms) to O(geoms).  Keep the old scene-per-env path for
    # heterogeneous playback models, which cannot share batched geometry.
    matching_models = all(
        model.ngeom == models[0].ngeom
        and np.array_equal(model.geom_type, models[0].geom_type)
        and np.array_equal(model.geom_size, models[0].geom_size)
        and np.array_equal(model.geom_dataid, models[0].geom_dataid)
        and np.array_equal(model.geom_rgba, models[0].geom_rgba)
        for model in models[1:]
    )
    if matching_models:
        return [
            {
                "batch": True,
                "runtime_env_indices": visible_env_indices.copy(),
                "models": models,
                "model": models[0],
                "data": data,
                "appliers": appliers,
                "scene": MujocoViserBatchScene(
                    server,
                    models,
                    name_prefix="/mujoco/batch",
                    position_offsets=np.column_stack(
                        (np.asarray(offsets, dtype=np.float64), np.zeros(len(offsets)))
                    ),
                    render_plane=True,
                ),
            }
        ]

    for local_idx, env_idx in enumerate(visible_env_indices):
        env_idx = int(env_idx)
        mj_model = models[local_idx]
        entries.append(
            {
                "slot_idx": local_idx,
                "runtime_env_idx": env_idx,
                "model": mj_model,
                "data": mujoco.MjData(mj_model),
                "applier": appliers[local_idx],
                "scene": MujocoViserScene(
                    server,
                    mj_model,
                    name_prefix=f"/mujoco/env_{local_idx}",
                    position_offset=_scene_offset(offsets[local_idx]),
                    render_plane=(local_idx == 0),
                ),
            }
        )
    return entries


def _close_scene_entries(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        entry["scene"].close()


def play_viser(args: PlayInteractiveArgs, cfg: DictConfig, *, algo: str = "ppo") -> None:
    if not VISER_AVAILABLE:
        raise SystemExit(
            "play_viser requires the viser extra. Install it with "
            "`pip install unilab[viser]` (or `uv sync --extra viser` in a source checkout)."
        )

    def log(message: str) -> None:
        print(f"[play_viser] {message}", flush=True)

    num_envs = int(OmegaConf.select(cfg, "viser.max_envs", default=16) or 16)
    session = create_playback_session(args, cfg, algo=algo, num_envs=num_envs, log=log)
    if session is None:
        return
    playback_session = session[0]
    env = playback_session.env
    assert_physics_state_playback_supported(env, entrypoint="play_viser")

    # --- GUI controls --------------------------------------------------------
    max_visible_envs = min(int(OmegaConf.select(cfg, "viser.max_envs", default=16) or 16), num_envs)
    env_options = tuple(f"env_{i}" for i in range(max_visible_envs))
    initial_env_idx = int(OmegaConf.select(cfg, "viser.env_idx", default=0) or 0)
    initial_env_idx = min(initial_env_idx, max_visible_envs - 1)
    initial_mode = str(OmegaConf.select(cfg, "viser.display_mode", default="all") or "all")
    if initial_mode not in {"single", "all"}:
        initial_mode = "all"
    visible_env_indices = build_visible_env_indices(num_envs, max_visible_envs)

    ctrl_dt = env.cfg.ctrl_dt
    render_spacing = float(
        OmegaConf.select(cfg, "training.render_spacing") or getattr(env.cfg, "render_spacing", 1.0)
    )

    # --- Setup viser server --------------------------------------------------
    port = int(OmegaConf.select(cfg, "viser.port", default=8080) or 8080)
    server = viser.ViserServer(port=port)

    with server.gui.add_folder("Controls"):
        display_dropdown = server.gui.add_dropdown(
            "Display",
            options=("all", "single"),
            initial_value=initial_mode,
        )
        env_dropdown = server.gui.add_dropdown(
            "Environment",
            options=env_options,
            initial_value=env_options[initial_env_idx],
        )
        pause_button = server.gui.add_button("Pause / Resume")
        step_button = server.gui.add_button("Step")
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )

    controls = PlaybackControls(
        paused=bool(getattr(args, "start_paused", False)),
        speed=float(getattr(args, "speed", 1.0)),
    )
    speed_slider.value = controls.speed
    env_idx = {"value": initial_env_idx}
    display_mode = {"value": initial_mode}
    scene_entries = {
        "value": _build_scene_entries(
            server,
            env,
            mode=display_mode["value"],
            selected_visible_idx=env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
    }

    def _rebuild_scenes() -> None:
        _close_scene_entries(scene_entries["value"])
        scene_entries["value"] = _build_scene_entries(
            server,
            env,
            mode=display_mode["value"],
            selected_visible_idx=env_idx["value"],
            visible_env_indices=visible_env_indices,
            spacing=render_spacing,
        )
        if display_mode["value"] == "single":
            runtime_idx = int(visible_env_indices[env_idx["value"]])
            log(f"Showing env_{env_idx['value']} (runtime env {runtime_idx})")
        else:
            log(
                f"Showing {max_visible_envs} env slots mapped to runtime envs "
                f"{visible_env_indices.tolist()}"
            )

    @pause_button.on_click
    def _on_pause_click(event: Any) -> None:
        del event
        paused = controls.toggle_pause()
        status = "paused" if paused else "resumed"
        log(status)

    @step_button.on_click
    def _on_step_click(event: Any) -> None:
        del event
        if not controls.paused:
            controls.pause()
            log("paused for single-step mode")
        controls.request_single_step()
        log("single step requested")

    @display_dropdown.on_update
    def _on_display_mode_update(event: Any) -> None:
        del event
        display_mode["value"] = str(display_dropdown.value)
        env_dropdown.disabled = display_mode["value"] == "all"
        _rebuild_scenes()

    @env_dropdown.on_update
    def _on_env_switch(event: Any) -> None:
        del event
        selected = env_dropdown.value
        idx = int(selected.split("_")[1])
        env_idx["value"] = idx
        if display_mode["value"] == "single":
            _rebuild_scenes()
        else:
            log(f"Selected env_{idx} (runtime env {int(visible_env_indices[idx])})")

    env_dropdown.disabled = display_mode["value"] == "all"

    playback_session.reset()

    log(f"Server running at http://localhost:{server.get_port()}")
    log(f"{num_envs} environment(s) loaded. Open browser to view.")
    if display_mode["value"] == "all":
        log(
            f"Rendering {max_visible_envs} env slots simultaneously from runtime envs "
            f"{visible_env_indices.tolist()}."
        )
    log("Press Ctrl+C to quit.")

    # --- Main loop -----------------------------------------------------------
    try:
        with torch.inference_mode():
            while True:
                t0 = time.perf_counter()

                controls.set_speed(float(speed_slider.value))
                playback_session.advance(controls)

                physics_batch = playback_session.physics_state()
                for entry in scene_entries["value"]:
                    if entry.get("batch", False):
                        for runtime_idx, applier, data in zip(
                            entry["runtime_env_indices"],
                            entry["appliers"],
                            entry["data"],
                            strict=True,
                        ):
                            applier.apply(physics_batch[int(runtime_idx)], data)
                        entry["scene"].update(entry["data"])
                        continue
                    entry["applier"].apply(
                        physics_batch[int(entry["runtime_env_idx"])], entry["data"]
                    )
                    entry["scene"].update(entry["data"])

                # Real-time pacing
                target_dt = controls.target_dt(ctrl_dt)
                elapsed = time.perf_counter() - t0
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[play_viser] Shutting down.")
    finally:
        _close_scene_entries(scene_entries["value"])


def main(argv: Sequence[str] | None = None) -> None:
    parsed = _parse_interactive_cli(sys.argv[1:] if argv is None else argv)
    cfg = _compose_interactive_config(parsed.algo, parsed.overrides)
    play_viser(_build_play_args(cfg, algo=parsed.algo), cfg, algo=parsed.algo)


if __name__ == "__main__":
    main()
