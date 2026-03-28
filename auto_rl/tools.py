from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .sandbox import SandboxPool, sandbox_exec

if TYPE_CHECKING:
    from .config import ProblemConfig

ALLOWED_COMMANDS = {"cp", "ls", "cat", "diff", "head", "tail", "wc", "grep"}


def create_tools(config: ProblemConfig, pool: SandboxPool) -> dict:
    """Create tool functions bound to config and pool. Returns name->func dict."""

    all_readable = set(config.readable_files) | set(config.editable_files)

    async def read_file(path: str, sandbox_id: str) -> str:
        """Read a file in the experiment sandbox.

        Args:
            path: File path relative to the challenge root.

        Returns:
            File content as a string, or an error message.
        """
        if path not in all_readable:
            return f"Error: cannot read '{path}'. Allowed: {sorted(all_readable)}"
        sandbox = pool.get(sandbox_id)
        file_path = sandbox.path / path
        try:
            return file_path.read_text()
        except FileNotFoundError:
            return f"Error: {path} not found"

    async def edit_file(path: str, content: str, sandbox_id: str) -> str:
        """Overwrite a file with new content (full file, not a diff).

        Args:
            path: File to edit (must be an editable file).
            content: The complete new file content.

        Returns:
            'OK' on success, or an error message.
        """
        if path not in config.editable_files:
            return f"Error: '{path}' is read-only. Editable: {sorted(config.editable_files)}"
        sandbox = pool.get(sandbox_id)
        (sandbox.path / path).write_text(content)
        return "OK"

    async def run_eval(sandbox_id: str) -> str:
        """Run the evaluation oracle and return the result.

        Returns:
            The metric result, or an error/crash message.
        """
        exit_code, stdout, stderr = await sandbox_exec(
            pool, sandbox_id, config.oracle_command, timeout=config.oracle_timeout
        )
        output = stdout + stderr
        if exit_code != 0:
            lines = output.strip().split("\n")
            tail = "\n".join(lines[-50:])
            return f"CRASH | last 50 lines:\n{tail}"
        m = re.search(config.metric_pattern, output)
        if m:
            value = m.group(1)
            return f"{config.metric_name}={value} | output:\n{output[-500:]}"
        return f"ERROR: could not parse {config.metric_name} from output:\n{output[-500:]}"

    async def run_command(command: str, sandbox_id: str) -> str:
        """Run a shell command in the sandbox directory.
        Allowed commands: cp, ls, cat, diff, head, tail, wc, grep.

        Args:
            command: The shell command to run.

        Returns:
            Command output, or an error message.
        """
        parts = command.strip().split()
        if not parts:
            return "Error: empty command"
        if parts[0] not in ALLOWED_COMMANDS:
            return f"Error: '{parts[0]}' not allowed. Allowed: {sorted(ALLOWED_COMMANDS)}"
        exit_code, stdout, stderr = await sandbox_exec(
            pool, sandbox_id, command, timeout=10
        )
        if exit_code != 0:
            return f"Error: {stderr or stdout}"
        return stdout

    return {
        "read_file": read_file,
        "edit_file": edit_file,
        "run_eval": run_eval,
        "run_command": run_command,
    }
