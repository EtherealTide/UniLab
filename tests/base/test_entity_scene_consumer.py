"""Typed scene authoring and public-only multi-entity manager transactions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from unisim.entities import EntityVariantBinding, SceneEntitySpec

from tests.envs.test_multi_entity_consumer import build_fixture_cfg
from unilab.base import backend_factory
from unilab.base.base import EnvCfg
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.scene import SceneCfg
from unilab.envs.manager_based_rl_env import make_manager_based_rl_env


def test_hydra_materializes_entity_sources_binding_and_immutable_assignment():
    scene = SceneCfg()
    apply_cfg_overrides(
        scene,
        OmegaConf.create(
            {
                "entity_assets": [
                    {
                        "name": "object",
                        "source": {"model_file": "a.xml"},
                        "kind": "rigid",
                        "initial_state": {"position": [1.0, 2.0, 3.0]},
                    }
                ],
                "entity_variant": {
                    "target_entity": "object",
                    "plan": {
                        "variants": [{"model_file": "a.xml"}, {"model_file": "b.xml"}],
                        "assignment": [1, 1, 0],
                        "layout": "same_layout",
                    },
                },
                "entities": {
                    "tool": {"root_body_name": "object/base", "physical_entity": "object"}
                },
            }
        ),
    )
    assert isinstance(scene.entity_assets[0], SceneEntitySpec)
    assert scene.entity_assets[0].initial_state.position == (1.0, 2.0, 3.0)
    assert isinstance(scene.entity_variant, EntityVariantBinding)
    np.testing.assert_array_equal(scene.entity_variant.plan.assignment, [1, 1, 0])
    assert not scene.entity_variant.plan.assignment.flags.writeable
    assert scene.entities["tool"].physical_entity == "object"
    with pytest.raises(ValueError):
        SceneCfg(model_file="legacy.xml", entity_assets=scene.entity_assets)


def test_asset_owner_sees_each_physical_and_variant_source_once(monkeypatch):
    cfg = build_fixture_cfg(passive=False)
    calls = []
    monkeypatch.setattr(
        backend_factory, "ensure_robot_assets_for_paths", lambda paths: calls.append(paths)
    )
    sentinel = object()
    monkeypatch.setattr(backend_factory.unisim, "create_backend", lambda *args, **kwargs: sentinel)
    assert backend_factory.create_backend("mujoco", cfg.scene, 2, 0.001) is sentinel
    assert len(calls) == 1
    expected = {entity.source.model_file for entity in cfg.scene.entity_assets if entity.source}
    expected.update(source.model_file for source in cfg.scene.entity_variant.plan.variants)
    assert expected.issubset(calls[0])
    assert len(calls[0]) == len(set(calls[0]))


@pytest.mark.parametrize("logical", ["robot", "object"])
def test_logical_root_cannot_point_to_a_physical_entity_descendant(logical):
    cfg = build_fixture_cfg(passive=True)
    child = "finger" if logical == "robot" else "lid"
    cfg.scene.entities[logical] = replace(
        cfg.scene.entities[logical], root_body_name=f"{logical}/{child}"
    )
    with pytest.raises(ValueError, match=f"physical root '{logical}/base'"):
        make_manager_based_rl_env(cfg, 2, "mujoco")


def test_hydra_constructs_new_scene_without_early_unisim_tuple_validation():
    cfg = EnvCfg()
    apply_cfg_overrides(
        cfg,
        OmegaConf.create(
            {
                "scene": {
                    "_target_": "unilab.base.scene.SceneCfg",
                    "entity_assets": [
                        {
                            "name": "object",
                            "kind": "rigid",
                            "source": {"model_file": "object.xml"},
                            "initial_state": {"quaternion": [1.0, 0.0, 0.0, 0.0]},
                        }
                    ],
                }
            }
        ),
    )
    assert isinstance(cfg.scene.entity_assets[0], SceneEntitySpec)
    assert cfg.scene.entity_assets[0].initial_state.quaternion == (1.0, 0.0, 0.0, 0.0)


def test_host_bridge_manager_entity_writes_fail_closed_before_runtime():
    with pytest.raises(NotImplementedError, match="mask_selected_reset"):
        make_manager_based_rl_env(build_fixture_cfg(passive=False), 2, "mujoco")


def test_host_bridge_manager_rejects_entity_row_runtime():
    with pytest.raises(NotImplementedError, match="mask_selected_reset"):
        make_manager_based_rl_env(build_fixture_cfg(passive=False), 2, "mujoco")


def test_per_environment_entity_defaults_are_not_broadcast_from_first_variant():
    cfg = build_fixture_cfg(passive=True)
    variants = cfg.scene.entity_variant.plan.variants
    for index, source in enumerate(variants):
        path = Path(source.model_file)
        text = path.read_text().replace(
            "</mujoco>",
            f'<keyframe><key name="home" qpos="0 0 1 1 0 0 0 {0.1 + 0.2 * index}"/></keyframe></mujoco>',
        )
        path.write_text(text)
    cfg.scene.default_keyframe_name = "home"
    # Every physical source needs the named key when selected by the composition owner.
    for entity in cfg.scene.entity_assets:
        if entity.name == "object":
            continue
        if entity.source is None:
            continue
        path = Path(entity.source.model_file)
        key = (
            '<key name="home" qpos="0.25" ctrl="0.37"/>'
            if entity.name == "robot"
            else '<key name="home"/>'
        )
        text = path.read_text().replace("</mujoco>", f"<keyframe>{key}</keyframe></mujoco>")
        path.write_text(text)
    with pytest.raises(NotImplementedError, match="mask_selected_reset"):
        make_manager_based_rl_env(cfg, 2, "mujoco")
