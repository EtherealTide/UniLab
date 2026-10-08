"""G1 motion profiles on the shared NumPy Manager-Based runtime."""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

G1_MOTION_TASKS = (
    "G1MotionTracking",
    "G1MotionTrackingSAC",
)

for _task_name in G1_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    if _task_name != "G1MotionTrackingSAC":
        registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")

# The canonical FlashSAC motion owners run the Manager tensor lifecycle.
registry.register_env(
    "G1MotionTrackingSAC",
    make_manager_based_rl_env,
    sim_backend="mujoco",
)
registry.register_env(
    "G1MotionTrackingSAC",
    make_manager_based_rl_env,
    sim_backend="mjwarp",
)
registry.register_env(
    "G1MotionTrackingSAC",
    make_manager_based_rl_env,
    sim_backend="genesis",
)
registry.register_env(
    "G1MotionTrackingSAC",
    make_manager_based_rl_env,
    sim_backend="newton",
)
registry.register_env(
    "G1MotionTrackingSAC",
    make_manager_based_rl_env,
    sim_backend="motrix",
)
__all__ = ["G1_MOTION_TASKS"]
