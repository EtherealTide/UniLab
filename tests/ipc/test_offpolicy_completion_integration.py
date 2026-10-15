"""Cross-package completion integration through the off-policy replay ingress."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf"


def test_eager_mujoco_completion_reaches_replay_ingress() -> None:
    """A public producer receipt must survive the worker-to-replay boundary."""
    pytest.importorskip(
        "unisim.backend.mujoco.backend",
        reason="unisim-core MuJoCo adapter (mjbatch build) not available",
    )
    pytest.importorskip("uni_rl.ipc.replay_buffer", reason="uni_rl is unavailable")
    from uni_rl.ipc.completion import ImmediateCompletion
    from uni_rl.ipc.replay_buffer import ReplayBuffer

    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=go2_joystick_flat/mujoco"])
    override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = registry.make(
        str(cfg.training.task_name),
        sim_backend="mujoco",
        env_cfg_override=override,
        num_envs=2,
    )
    replay = ReplayBuffer(
        capacity=16,
        obs_dim=env.obs_groups_spec["obs"],
        action_dim=env.action_space.shape[0],
        device="cpu",
        ingress_slot_rows=2,
        critic_dim=env.obs_groups_spec["critic"],
    )

    try:
        initial_obs, _ = env.reset(seed=2359)
        current_obs = initial_obs
        states = []
        packed_rows: list[torch.Tensor] = []
        source_snapshots: list[dict[str, torch.Tensor]] = []
        next_snapshots: list[dict[str, torch.Tensor]] = []

        for tick in range(5):
            source_snapshots.append({name: value.clone() for name, value in current_obs.items()})
            actions = torch.full(
                (2, env.action_space.shape[0]),
                (tick + 1) * 0.02,
                dtype=torch.float32,
                device=env.device,
            )
            state = env.step(actions)
            states.append(state)
            next_snapshots.append({name: value.clone() for name, value in state.obs.items()})

            completion = state.info.get("completion")
            assert completion is not None
            assert completion.__class__.__module__.startswith("unisim.")

            done = state.terminated | state.truncated
            terminal_final = state.final_observation
            terminal_obs = terminal_final.get("obs") if terminal_final is not None else None
            terminal_critic = terminal_final.get("critic") if terminal_final is not None else None
            assert replay.add_batch(
                current_obs["obs"],
                actions,
                state.reward.to(dtype=torch.float32).ravel(),
                state.obs["obs"],
                done.to(dtype=torch.float32),
                state.truncated.to(dtype=torch.float32),
                terminal_mask=done,
                terminal_next_obs=terminal_obs,
                critic=current_obs["critic"],
                next_critic=state.obs["critic"],
                terminal_next_critic=terminal_critic,
                completion=completion,
            )

            published = replay.last_completion
            assert published is not completion
            assert isinstance(published, ImmediateCompletion)
            assert published.contract_version == completion.contract_version
            assert published.operation == completion.operation
            assert published.sequence == completion.sequence
            assert published.stream_id == completion.stream_id
            assert published.producer_id == completion.producer_id

            ingress = replay.take_published_ingress()
            assert ingress is not None
            slot, start, count, packed = ingress
            replay.commit_ingress(slot=slot, start=start, count=count)
            assert (start, count) == (tick * 2, 2)
            packed_rows.append(packed.clone())
            current_obs = state.obs

        # A public observation remains valid after later eager steps even when
        # the consumer and replay publication retain it without an extra clone.
        for tick, (state, packed) in enumerate(zip(states, packed_rows, strict=True)):
            for group, field in (("obs", "obs"), ("critic", "critic")):
                torch.testing.assert_close(
                    replay.field_view(packed, field),
                    source_snapshots[tick][group],
                )
            for group, field in (("obs", "next_obs"), ("critic", "next_critic")):
                expected = next_snapshots[tick][group]
                torch.testing.assert_close(state.obs[group], expected)
                torch.testing.assert_close(replay.field_view(packed, field), expected)
    finally:
        replay.close()
        env.close()
