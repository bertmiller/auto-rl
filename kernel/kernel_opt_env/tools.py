"""
Tools exposed to the RL agent during kernel optimization episodes.

All tools are async Python functions. `sandbox_id` is injected by
StatefulToolEnv via update_tool_args and hidden from the model's
schema via args_to_skip.
"""

from __future__ import annotations

from .sandbox import (
    sandbox_read_file,
    sandbox_run_analysis,
    sandbox_run_command,
    sandbox_run_tests,
    sandbox_write_file,
)

# Paths the agent is allowed to read
READABLE_PATHS = {
    "perf_takehome.py",
    "problem.py",
    "test_output.log",
    "tests/submission_tests.py",
    "best.py",
}


async def read_file(
    path: str,
    sandbox_id: str,
) -> str:
    """
    Read the current content of a file in the experiment sandbox.

    Args:
        path: File path relative to the challenge root.
              Use 'perf_takehome.py' to read the kernel code,
              'problem.py' to read the simulator and ISA reference,
              'test_output.log' to read the last test run output.

    Returns:
        Full file content as a string.
    """
    if path not in READABLE_PATHS:
        return f"Error: cannot read '{path}'. Allowed: {sorted(READABLE_PATHS)}"
    return await sandbox_read_file(sandbox_id, path)


async def edit_file(
    path: str,
    content: str,
    sandbox_id: str,
) -> str:
    """
    Overwrite perf_takehome.py with new content.
    Only perf_takehome.py may be edited. problem.py and tests/ are read-only.

    Args:
        path: Must be 'perf_takehome.py'.
        content: The complete new file content. Full file, not a diff.

    Returns:
        'OK' on success, or an error message if path is restricted.
    """
    if path != "perf_takehome.py":
        return f"Error: {path} is read-only. Only perf_takehome.py may be edited."
    await sandbox_write_file(sandbox_id, path, content)
    return "OK"


def _format_test_result(result: dict) -> str:
    if result["crashed"]:
        return f"CRASH | last 50 lines:\n{result['output']}"
    if not result["correct"]:
        return f"INCORRECT | cycles={result['cycles']} | output:\n{result['output'][-500:]}"
    return (
        f"CORRECT | cycles={result['cycles']} | "
        f"speedup={result['speedup']}x | "
        f"tests_passed={result['tests_passed']}/{result['total_tests']}"
    )


async def run_tests(
    sandbox_id: str,
) -> str:
    """
    Run the submission tests and return correctness + cycle count.
    Uses the frozen simulator in tests/frozen_problem.py to prevent
    cheating by modifying the simulator.

    Returns:
        Success: "CORRECT | cycles=18532 | speedup=7.97x | tests_passed=3/8"
        Failure: "INCORRECT | cycles=... | output: ..."
        Crash:   "CRASH | last 50 lines: ..."
    """
    result = await sandbox_run_tests(sandbox_id)
    return _format_test_result(result)


def _format_analysis_result(result: dict) -> str:
    if not result["success"]:
        return f"ERROR:\n{result['output']}"
    return result["output"]


async def run_analysis(
    script: str,
    sandbox_id: str,
) -> str:
    """
    Run an inline Python analysis script to inspect the kernel's behavior.
    The script executes in the sandbox with access to problem.py and
    perf_takehome.py imports. Use this to analyze bottlenecks, inspect
    instruction schedules, count slot utilization, profile cycle
    distribution, etc.

    The script runs with a 30-second timeout.

    Args:
        script: A complete Python script as a string. It can import from
                problem.py and perf_takehome.py. Print results to stdout.

    Returns:
        stdout output (truncated to 10,000 chars), or an error message
        if the script crashes or times out.
    """
    result = await sandbox_run_analysis(sandbox_id, script, timeout_secs=30)
    return _format_analysis_result(result)


def _format_command_result(result: dict) -> str:
    if not result["success"]:
        return result["output"]
    return result["output"]


async def run_command(
    command: str,
    sandbox_id: str,
) -> str:
    """
    Run a shell command in the sandbox directory.
    Only these commands are allowed: cp, ls, cat, diff, head, tail, wc, grep.

    Use this for file management (e.g. checkpointing with
    'cp perf_takehome.py best.py') and quick inspection.

    Args:
        command: The shell command to run.

    Returns:
        Command output (stdout), or an error message.
    """
    result = await sandbox_run_command(sandbox_id, command, timeout_secs=10)
    return _format_command_result(result)
