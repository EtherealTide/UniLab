from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.g1 import flashsac_owner_contract as module

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"


def _materialize_task(task: str, *, algo: str = "flashsac") -> ManagerBasedRlEnvCfg:
    GlobalHydra.instance().clear()
    config_dir = CONF_DIR if algo == "flashsac" else ROOT_DIR / "src/unilab/conf/sac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(
        composed, root_dir=ROOT_DIR, algo_name=algo
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def test_flashsac_owner_fingerprint_accepts_canonical_backends() -> None:
    mujoco = _materialize_task("g1_motion_tracking/mujoco")
    mjwarp = _materialize_task("g1_motion_tracking/mjwarp")
    expected = module._TORCH_G1_MANAGER_TERMS_OWNER_IDENTITY_V1
    assert module._torch_g1_flashsac_owner_identity(mujoco) == expected
    assert module._torch_g1_flashsac_owner_identity(mjwarp) == expected


def test_flashsac_motrix_owner_uses_generic_manager_runtime() -> None:
    registry.ensure_registries()
    assert "motrix" in registry._envs["G1MotionTracking"].env_factory_dict


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.observations["actor"].terms["base_lin_vel"].noise, "n_min", -0.2),
        lambda cfg: cfg.rewards["motion_global_root_pos"].params.__setitem__("std", 0.4),
        lambda cfg: setattr(cfg.actions["joint_pos"], "clip", {"joint": (0.0, 1.0)}),
        lambda cfg: setattr(cfg.commands["motion"].params, "adaptive_alpha", 0.1),
    ],
)
def test_flashsac_owner_fingerprint_fails_closed(mutate) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    mutate(cfg)
    with pytest.raises(ValueError, match="canonical owner contract"):
        module._validate_torch_g1_flashsac_owner_contract(cfg)


def test_motion_reward_manager_terms_share_owner_identity() -> None:
    for backend in ("mujoco", "mjwarp", "newton"):
        cfg = _materialize_task(f"g1_motion_tracking/{backend}")
        identity = module._torch_g1_flashsac_owner_identity(cfg)
        assert identity == module._TORCH_G1_MANAGER_TERMS_OWNER_IDENTITY_V1


def test_motion_reward_terms_stay_contract_aligned() -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")

    expected = (
        "motion_global_root_pos",
        "motion_global_root_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    )
    for name in expected:
        assert name in cfg.rewards


def test_flashsac_owner_rejects_fixed_model_variants_before_backend_creation() -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    cfg.fixed_model_variants = FixedModelVariantCatalogCfg(
        variants=(FixedModelVariantCfg(name="variant", source_model_file="variant.xml"),)
    )
    with pytest.raises(ValueError, match="does not support fixed model variants"):
        module._validate_torch_g1_flashsac_owner_contract(cfg)
