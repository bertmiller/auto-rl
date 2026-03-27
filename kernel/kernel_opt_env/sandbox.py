"""
Sandbox layer for kernel-opt RL environment.

Each sandbox is an isolated directory copy of original_performance_takehome/.
No GPU needed — the oracle is a pure Python simulator.

Pre-provisioning: none required. Sandboxes are created automatically
by copying the challenge directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
CHALLENGE_DIR = Path(
    os.environ.get(
        "KERNEL_OPT_CHALLENGE_DIR",
        str(_THIS_DIR.parent / "original_performance_takehome"),
    )
)
SANDBOXES_DIR = Path(
    os.environ.get("KERNEL_OPT_SANDBOXES_DIR", "/tmp/kernel-opt-sandboxes")
)
ARTIFACTS_DIR = Path(
    os.environ.get("KERNEL_OPT_ARTIFACTS_DIR", str(_THIS_DIR.parent / "artifacts"))
)
STARTING_POINTS_DIR = Path(
    os.environ.get(
        "KERNEL_OPT_STARTING_POINTS_DIR",
        str(_THIS_DIR.parent / "starting_points"),
    )
)

# Files that constitute the clean challenge state
CHALLENGE_FILES = [
    "perf_takehome.py",
    "problem.py",
    "watch_trace.py",
    "watch_trace.html",
]
CHALLENGE_DIRS = [
    "tests",
]

# Commands the agent is allowed to run via run_command
ALLOWED_COMMANDS = {"cp", "ls", "cat", "diff", "head", "tail", "wc", "grep"}


@dataclass
class Sandbox:
    id: str
    path: Path


class KernelOptSandboxPool:
    """
    Manages a pool of isolated directory sandboxes.
    Sandboxes are reused across episodes after reset.
    """

    def __init__(self, size: int = 4, **kwargs):
        self.size = size
        self._available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self._all: dict[str, Sandbox] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        SANDBOXES_DIR.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        for i in range(self.size):
            sandbox = await loop.run_in_executor(None, self._create_sandbox, i)
            self._all[sandbox.id] = sandbox
            await self._available.put(sandbox)
        self._initialized = True
        log.info(
            "Sandbox pool initialized: %d sandboxes from %s",
            self.size,
            CHALLENGE_DIR,
        )

    def _create_sandbox(self, index: int) -> Sandbox:
        sid = f"sandbox-{index}"
        sandbox_path = SANDBOXES_DIR / sid
        if sandbox_path.exists():
            self._reset_sandbox_dir(sandbox_path)
        else:
            self._copy_challenge(sandbox_path)
        log.info("Created sandbox %s at %s", sid, sandbox_path)
        return Sandbox(id=sid, path=sandbox_path)

    def _copy_challenge(self, dest: Path) -> None:
        """Copy challenge files into a new sandbox directory."""
        dest.mkdir(parents=True, exist_ok=True)
        for filename in CHALLENGE_FILES:
            src = CHALLENGE_DIR / filename
            if src.exists():
                shutil.copy2(src, dest / filename)
        for dirname in CHALLENGE_DIRS:
            src = CHALLENGE_DIR / dirname
            if src.exists():
                shutil.copytree(src, dest / dirname, dirs_exist_ok=True)

    def _reset_sandbox_dir(self, path: Path) -> None:
        """Reset sandbox to clean challenge state."""
        # Remove agent-created files (best.py, analysis scripts, logs, traces)
        for item in path.iterdir():
            if item.name in ("tests",):
                continue
            if item.name not in CHALLENGE_FILES:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        # Restore challenge files from source
        for filename in CHALLENGE_FILES:
            src = CHALLENGE_DIR / filename
            dst = path / filename
            if src.exists():
                shutil.copy2(src, dst)

    async def acquire(self) -> str:
        if not self._initialized:
            await self.initialize()
        sandbox = await self._available.get()
        return sandbox.id

    async def release(self, sandbox_id: str) -> None:
        sandbox = self._all.get(sandbox_id)
        if sandbox is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._reset_sandbox_dir, sandbox.path)
        await self._available.put(sandbox)

    async def shutdown(self) -> None:
        log.info("Sandbox pool shut down")


# --- Module-level pool reference (set by env.py) ---

_pool: KernelOptSandboxPool | None = None


def _sandbox(sandbox_id: str) -> Sandbox:
    assert _pool is not None, "Sandbox pool not initialized"
    return _pool._all[sandbox_id]


# --- Subprocess execution ---


async def _exec(
    sandbox_id: str, cmd: str, timeout: int = 60
) -> tuple[int, str, str]:
    """Execute a shell command in the sandbox directory. Returns (exit_code, stdout, stderr)."""
    sandbox = _sandbox(sandbox_id)

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                cwd=str(sandbox.path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 1, "", f"Command timed out after {timeout}s: {cmd[:100]}"

    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    err = stderr.decode("utf-8", errors="replace") if stderr else ""
    return proc.returncode or 0, out, err


# --- Sandbox operations (called by tools) ---


async def sandbox_install_variant(sandbox_id: str, variant_file: str) -> None:
    """Copy a starting-point perf_takehome.py variant into the sandbox."""
    sandbox = _sandbox(sandbox_id)
    src = STARTING_POINTS_DIR / variant_file
    dst = sandbox.path / "perf_takehome.py"
    shutil.copy2(src, dst)


async def sandbox_read_file(sandbox_id: str, path: str) -> str:
    sandbox = _sandbox(sandbox_id)
    file_path = sandbox.path / path
    try:
        return file_path.read_text()
    except FileNotFoundError:
        return f"Error: {path} not found"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def sandbox_write_file(sandbox_id: str, path: str, content: str) -> None:
    sandbox = _sandbox(sandbox_id)
    file_path = sandbox.path / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


async def sandbox_run_tests(sandbox_id: str) -> dict:
    """Run submission_tests.py and return parsed results."""
    import re

    cmd = "python3 tests/submission_tests.py"
    exit_code, stdout, stderr = await _exec(sandbox_id, cmd, timeout=120)
    output = stdout + stderr

    # Write test output log for the agent to read
    await sandbox_write_file(sandbox_id, "test_output.log", output)

    if exit_code != 0 and "CYCLES:" not in output:
        lines = output.strip().split("\n")
        tail = "\n".join(lines[-50:])
        return {"crashed": True, "correct": False, "output": tail}

    # Parse cycle count from output
    cycles = None
    for line in output.split("\n"):
        if m := re.search(r"CYCLES:\s*(\d+)", line):
            cycles = int(m.group(1))

    if cycles is None:
        return {"crashed": True, "correct": False, "output": "No cycle count found.\n" + output[-500:]}

    # Check correctness: did the CorrectnessTests pass?
    # Speed test failures show as "FAIL" but correctness failures show "Incorrect output values"
    # or an exception/error in the correctness test class itself.
    correct = "Incorrect output values" not in output and "ERROR: test_kernel_correctness" not in output

    # Count speed tests passed
    speed_tests = [
        ("test_kernel_speedup", 147734),
        ("test_kernel_updated_starting_point", 18532),
        ("test_opus4_many_hours", 2164),
        ("test_opus45_casual", 1790),
        ("test_opus45_2hr", 1579),
        ("test_sonnet45_many_hours", 1548),
        ("test_opus45_11hr", 1487),
        ("test_opus45_improved_harness", 1363),
    ]
    tests_passed = sum(1 for _, threshold in speed_tests if cycles < threshold)

    return {
        "crashed": False,
        "correct": correct,
        "cycles": cycles,
        "speedup": round(147734 / cycles, 2) if cycles > 0 else 0,
        "tests_passed": tests_passed,
        "total_tests": len(speed_tests),
        "output": output,
    }


async def sandbox_run_analysis(
    sandbox_id: str, script: str, timeout_secs: int = 30
) -> dict:
    """Write script to temp file in sandbox and execute it."""
    sandbox = _sandbox(sandbox_id)
    script_name = f"_analysis_{uuid.uuid4().hex[:8]}.py"
    script_path = sandbox.path / script_name

    try:
        script_path.write_text(script)
        exit_code, stdout, stderr = await _exec(
            sandbox_id, f"python3 {script_name}", timeout=timeout_secs
        )

        if exit_code != 0:
            return {"success": False, "output": stderr or stdout}

        # Truncate stdout to 10k chars
        if len(stdout) > 10000:
            stdout = stdout[:10000] + "\n... (truncated)"

        return {"success": True, "output": stdout}
    finally:
        if script_path.exists():
            script_path.unlink()


async def sandbox_run_command(
    sandbox_id: str, command: str, timeout_secs: int = 10
) -> dict:
    """Run an allowed shell command in the sandbox."""
    parts = command.strip().split()
    if not parts:
        return {"success": False, "output": "Error: empty command"}

    cmd_name = parts[0]
    if cmd_name not in ALLOWED_COMMANDS:
        return {
            "success": False,
            "output": f"Error: '{cmd_name}' not allowed. Allowed: {sorted(ALLOWED_COMMANDS)}",
        }

    exit_code, stdout, stderr = await _exec(sandbox_id, command, timeout=timeout_secs)

    if exit_code != 0:
        return {"success": False, "output": stderr or stdout or f"Command failed with exit code {exit_code}"}

    return {"success": True, "output": stdout}


async def sandbox_export_artifacts(
    sandbox_id: str, episode_id: str, state: dict
) -> Path:
    """Export episode artifacts from sandbox before it's reset."""
    sandbox = _sandbox(sandbox_id)
    episode_dir = ARTIFACTS_DIR / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["perf_takehome.py", "best.py", "test_output.log"]:
        src = sandbox.path / filename
        if src.exists():
            shutil.copy2(src, episode_dir / filename)

    history = state.get("optimization_history", [])
    summary = {
        "episode_id": episode_id,
        "sandbox_id": sandbox_id,
        "baseline_cycles": state.get("baseline_cycles"),
        "best_cycles": state.get("best_cycles"),
        "speedup": round(
            state.get("baseline_cycles", 147734)
            / max(state.get("best_cycles", 147734), 1),
            2,
        ),
        "num_attempts": state.get("optimization_step", 0),
        "num_failures": state.get("num_failures", 0),
        "optimization_history": history,
    }
    (episode_dir / "episode.json").write_text(json.dumps(summary, indent=2))

    log.info("Exported artifacts for episode %s to %s", episode_id, episode_dir)
    return episode_dir
