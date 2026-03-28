"""
Tools exposed to the RL agent during autoresearch episodes.

All tools are async Python functions. `sandbox_id` is injected by
StatefulToolEnv via update_tool_args and hidden from the model's
schema via args_to_skip.
"""

from __future__ import annotations

import re

from .sandbox import (
    sandbox_git_commit,
    sandbox_git_revert,
    sandbox_read_file,
    sandbox_run_train,
    sandbox_write_file,
)


async def read_file(
    path: str,
    sandbox_id: str,
) -> str:
    """
    Read the current content of a file in the experiment sandbox.

    Args:
        path: File path relative to the autoresearch repo root.
              Use 'train.py' to read the training script,
              'run.log' to read the last experiment output,
              'results.tsv' to read the experiment history.

    Returns:
        Full file content as a string.
    """
    return await sandbox_read_file(sandbox_id, path)


async def edit_file(
    path: str,
    content: str,
    sandbox_id: str,
) -> str:
    """
    Overwrite train.py with new content.
    Only train.py may be edited. prepare.py and program.md are read-only.

    Args:
        path: Must be 'train.py'.
        content: The complete new file content. Full file, not a diff.

    Returns:
        'OK' on success, or an error message if path is restricted.
    """
    if path != "train.py":
        return f"Error: {path} is read-only. Only train.py may be edited."
    await sandbox_write_file(sandbox_id, path, content)
    return "OK"


def _format_run_result(result: dict) -> str:
    if result["crashed"]:
        return f"CRASH | last 50 lines of traceback:\n{result['output']}"
    return (
        f"val_bpb={result['val_bpb']:.4f} | "
        f"peak_vram_mb={result['peak_vram_mb']} | "
        f"training_seconds={result['training_seconds']:.1f}"
    )


async def run_experiment(
    sandbox_id: str,
) -> str:
    """
    Run the training experiment and return the result.
    Training runs for a fixed 5-minute wall-clock budget.

    Returns:
        Success: "val_bpb=0.9847 | peak_vram_mb=44210 | training_seconds=300.1"
        Crash:   "CRASH | last 50 lines of traceback: ..."
    """
    result = await sandbox_run_train(sandbox_id, train_secs=300)
    return _format_run_result(result)


async def git_commit_or_revert(
    action: str,
    message: str = "",
    sandbox_id: str = "",
) -> str:
    """
    Commit or revert the current train.py state.

    Args:
        action: 'commit' to keep this change, 'revert' to discard it.
        message: Commit message if action='commit'. Should describe
                 what changed and the val_bpb result.

    Returns:
        Confirmation string with new git HEAD or revert status.
    """
    if action == "commit":
        sha = await sandbox_git_commit(sandbox_id, message)
        return f"Committed: {sha[:7]}"
    elif action == "revert":
        await sandbox_git_revert(sandbox_id)
        return "Reverted to last committed state."
    else:
        return f"Error: action must be 'commit' or 'revert', got '{action}'"
