import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src" / "unilab"


def test_reset_chain_does_not_materialize_dynamic_done_rows() -> None:
    guarded_files = (
        "envs/manager_based_rl_env.py",
        "managers/event_manager.py",
        "managers/command_manager.py",
        "managers/observation_manager.py",
        "rl/vec_env.py",
    )
    forbidden = (
        "reset_buf.nonzero",
        "terminated.nonzero",
        "truncated.nonzero",
        "dones.nonzero",
        "done.nonzero",
    )
    for relative_path in guarded_files:
        source = (SOURCE_ROOT / relative_path).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), relative_path


def test_removed_reset_row_contracts_stay_removed() -> None:
    removed = (
        "uses_tensor_reset_rows",
        "reset_command_state",
        "compute_reset_rows",
        "reset_to_default_mask",
        "write_root_state_tensor_to_sim_mask",
        "write_motion_state_tensor_to_sim_mask",
        "commit_tensor(",
        "can_commit_packed",
        "refresh_after_selected_reset",
    )
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        identifiers.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        identifiers.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        assert not identifiers.intersection(removed), path
