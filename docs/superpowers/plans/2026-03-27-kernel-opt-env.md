# kernel-opt-env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the kernel optimization RL environment so models can be trained to optimize VLIW SIMD kernels via `verifiers` + `prime-rl`.

**Architecture:** Mirror the nanochat `autoresearch_env` package structure. Five modules: sandbox pool (directory-copy isolation), tools (5 agent-facing tools), rubric (log2 speedup reward), env (StatefulToolEnv subclass), and entrypoint (__init__.py with dataset + load_environment). No GPU needed — the oracle is a pure Python simulator.

**Tech Stack:** Python 3.10+, `verifiers` framework, `datasets` library, `asyncio` subprocess execution.

**Reference implementation:** `nanochat/autoresearch_env/` — follow the same patterns for tool registration, state management, env_response post-processing, cleanup, and artifact export.

---

## File Structure

```
kernel/
├── spec.md                              # exists
├── original_performance_takehome/       # exists, read-only challenge source
├── pyproject.toml                       # NEW — package definition
├── rl.toml                              # NEW — prime-rl config
├── .gitignore                           # NEW
└── kernel_opt_env/                      # NEW — all 5 modules below
    ├── __init__.py
    ├── sandbox.py
    ├── tools.py
    ├── rubric.py
    └── env.py
```

---

### Task 1: Package scaffolding

**Files:**
- Create: `kernel/pyproject.toml`
- Create: `kernel/.gitignore`
- Create: `kernel/kernel_opt_env/__init__.py` (empty placeholder)

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "kernel-opt-env"
version = "0.1.0"
description = "RL environment for training models to optimize VLIW SIMD kernels"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.0",
    "datasets",
]

[project.entry-points."verifiers.envs"]
kernel-opt = "kernel_opt_env:load_environment"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
*.log
/artifacts/
```

- [ ] **Step 3: Create empty __init__.py**

```python
"""
kernel-opt-env: RL environment for VLIW SIMD kernel optimization.
"""
```

- [ ] **Step 4: Verify structure**

Run: `ls -la kernel/kernel_opt_env/ && cat kernel/pyproject.toml`
Expected: __init__.py exists, pyproject.toml is valid.

- [ ] **Step 5: Commit**

```bash
git add kernel/pyproject.toml kernel/.gitignore kernel/kernel_opt_env/__init__.py
git commit -m "feat(kernel): scaffold kernel-opt-env package"
```

---

### Task 2: Sandbox pool

**Files:**
- Create: `kernel/kernel_opt_env/sandbox.py`

- [ ] **Step 1: Write sandbox.py**

```python
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
from dataclasses import dataclass, field
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
            if item.name not in [f for f in CHALLENGE_FILES]:
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
    cmd = "python tests/submission_tests.py 2>&1"
    exit_code, stdout, stderr = await _exec(sandbox_id, cmd, timeout=120)
    output = stdout + stderr

    # Write test output log for the agent to read
    await sandbox_write_file(sandbox_id, "test_output.log", output)

    if exit_code != 0 and "CYCLES:" not in output:
        lines = output.strip().split("\n")
        tail = "\n".join(lines[-50:])
        return {"crashed": True, "correct": False, "output": tail}

    # Parse cycle count from output
    import re

    cycles = None
    for line in output.split("\n"):
        if m := re.search(r"CYCLES:\s*(\d+)", line):
            cycles = int(m.group(1))

    if cycles is None:
        return {"crashed": True, "correct": False, "output": "No cycle count found.\n" + output[-500:]}

    # Check correctness: did the correctness tests pass?
    correct = "Incorrect" not in output and "FAIL" not in output and "Error" not in output

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
            sandbox_id, f"python {script_name}", timeout=timeout_secs
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
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/robert/repos/auto-rl/kernel && python -c "import ast; ast.parse(open('kernel_opt_env/sandbox.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add kernel/kernel_opt_env/sandbox.py
git commit -m "feat(kernel): add sandbox pool with subprocess execution"
```

---

### Task 3: Tools

**Files:**
- Create: `kernel/kernel_opt_env/tools.py`

- [ ] **Step 1: Write tools.py**

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/robert/repos/auto-rl/kernel && python -c "import ast; ast.parse(open('kernel_opt_env/tools.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add kernel/kernel_opt_env/tools.py
git commit -m "feat(kernel): add agent tools (read, edit, tests, analysis, command)"
```

---

### Task 4: Rubric

**Files:**
- Create: `kernel/kernel_opt_env/rubric.py`

- [ ] **Step 1: Write rubric.py**

```python
"""
Reward functions for the kernel optimization RL environment.

Primary reward: log2 speedup over baseline (147,734 cycles).
Process reward: fraction of attempts that set a new best.
"""

from __future__ import annotations

import math

import verifiers as vf


BASELINE_CYCLES = 147734


async def speedup_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Primary reward: log2 speedup over baseline.

    Reference points:
      - 2x speedup (73867 cycles)  -> reward 1.0
      - 8x speedup (18467 cycles)  -> reward 3.0
      - 100x speedup (1477 cycles) -> reward 6.6

    Failure penalty: 0.05 per failure subtracted.
    """
    baseline = state.get("baseline_cycles", BASELINE_CYCLES)
    best = state.get("best_cycles", baseline)
    failure_penalty = state.get("num_failures", 0) * 0.05

    if best >= baseline:
        return max(0.0, -failure_penalty)

    raw = math.log2(baseline / best)
    return float(max(0.0, min(raw - failure_penalty, 10.0)))


async def monotonic_improvement_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Process reward: fraction of non-baseline, correct attempts
    that set a new best cycle count.

    Rewards systematic optimization over random thrashing.
    """
    history = [
        e for e in state.get("optimization_history", [])
        if e["correct"] and e["step"] > 0
    ]
    if not history:
        return 0.0

    improvements = 0
    best_so_far = state.get("baseline_cycles", BASELINE_CYCLES)
    for e in history:
        if e["cycles"] < best_so_far:
            improvements += 1
            best_so_far = e["cycles"]

    return improvements / len(history)


rubric = vf.Rubric(
    funcs=[
        speedup_reward,
        monotonic_improvement_reward,
    ],
    weights=[1.0, 0.25],
)
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/robert/repos/auto-rl/kernel && python -c "import ast; ast.parse(open('kernel_opt_env/rubric.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add kernel/kernel_opt_env/rubric.py
git commit -m "feat(kernel): add reward functions (log2 speedup + monotonic improvement)"
```

---

### Task 5: Environment class

**Files:**
- Create: `kernel/kernel_opt_env/env.py`

- [ ] **Step 1: Write env.py**

```python
"""
KernelOptEnv: RL environment for training models to optimize
VLIW SIMD kernels by iterative code editing.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

import verifiers as vf
from verifiers.types import Messages, State

from .rubric import rubric, BASELINE_CYCLES
from .sandbox import (
    KernelOptSandboxPool,
    sandbox_export_artifacts,
)
from .tools import edit_file, read_file, run_analysis, run_command, run_tests
from . import sandbox as sandbox_mod

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class KernelOptEnv(vf.StatefulToolEnv):
    """
    RL environment for training models to optimize kernels on a custom
    VLIW SIMD architecture.

    One episode = one optimization session. The agent iterates on
    perf_takehome.py for up to max_attempts optimization cycles,
    starting from the same naive baseline every time.

    Per-rollout state:
        sandbox_id: str               -- isolated working directory
        baseline_cycles: int           -- 147734 (naive scalar implementation)
        best_cycles: int               -- best cycle count this episode
        current_cycles: int            -- most recent cycle count
        optimization_history: list     -- [{step, cycles, speedup, correct, description}]
        num_failures: int              -- incorrect results or crashes
        optimization_step: int         -- current step counter
        episode_done: bool
    """

    def __init__(
        self,
        max_attempts: int = 20,
        sandbox_pool_size: int = 4,
        **kwargs,
    ):
        super().__init__(
            rubric=rubric,
            max_turns=max_attempts * 4,
            **kwargs,
        )
        self.max_attempts = max_attempts
        self.sandbox_pool = KernelOptSandboxPool(size=sandbox_pool_size)
        # Set module-level pool reference for sandbox operations
        sandbox_mod._pool = self.sandbox_pool

        # Register tools with sandbox_id hidden from model schema
        self.add_tool(read_file, args_to_skip=["sandbox_id"])
        self.add_tool(edit_file, args_to_skip=["sandbox_id"])
        self.add_tool(run_tests, args_to_skip=["sandbox_id"])
        self.add_tool(run_analysis, args_to_skip=["sandbox_id"])
        self.add_tool(run_command, args_to_skip=["sandbox_id"])

    async def setup_state(self, state: State, **kwargs) -> State:
        """
        Allocate a fresh sandbox with a clean copy of the challenge files.
        No baseline run needed -- the baseline cycle count is a known constant.
        """
        sandbox_id = await self.sandbox_pool.acquire()

        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        state.update({
            "sandbox_id": sandbox_id,
            "episode_id": episode_id,
            "baseline_cycles": BASELINE_CYCLES,
            "best_cycles": BASELINE_CYCLES,
            "current_cycles": BASELINE_CYCLES,
            "optimization_history": [{
                "step": 0,
                "cycles": BASELINE_CYCLES,
                "speedup": 1.0,
                "correct": True,
                "description": "baseline (naive scalar, no ILP, no SIMD)",
            }],
            "num_failures": 0,
            "optimization_step": 0,
            "episode_done": False,
        })

        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages, state: State, **kwargs
    ) -> dict:
        """Inject sandbox_id into all tool calls."""
        updated = dict(tool_args)
        updated["sandbox_id"] = state["sandbox_id"]
        return updated

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Dispatch tool calls via parent, then post-process run_tests
        results to track optimization state.
        """
        # Collect tool_call_id -> tool_name mapping from the assistant message
        run_tests_call_ids: set[str] = set()
        last_msg = messages[-1] if messages else {}
        tool_calls = (
            last_msg.get("tool_calls", [])
            if isinstance(last_msg, dict)
            else getattr(last_msg, "tool_calls", []) or []
        )
        for tc in tool_calls:
            fn = (
                tc.get("function", {})
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            name = (
                fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            )
            tc_id = (
                tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            )
            if name == "run_tests":
                run_tests_call_ids.add(tc_id)

        tool_messages = await super().env_response(messages, state, **kwargs)

        # Post-process only run_tests tool responses
        for msg in tool_messages:
            tc_id = (
                msg.get("tool_call_id", "")
                if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            if tc_id not in run_tests_call_ids:
                continue

            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            if not content:
                continue

            if content.startswith("CORRECT"):
                m = re.search(r"cycles=(\d+)", content)
                if m:
                    cycles = int(m.group(1))
                    state["optimization_step"] = state.get("optimization_step", 0) + 1
                    step = state["optimization_step"]
                    state["current_cycles"] = cycles

                    if cycles < state["best_cycles"]:
                        state["best_cycles"] = cycles

                    state["optimization_history"].append({
                        "step": step,
                        "cycles": cycles,
                        "speedup": round(BASELINE_CYCLES / cycles, 2),
                        "correct": True,
                        "description": "",
                    })

                    if step >= self.max_attempts:
                        state["episode_done"] = True

            elif content.startswith("INCORRECT") or content.startswith("CRASH"):
                state["num_failures"] = state.get("num_failures", 0) + 1
                state["optimization_step"] = state.get("optimization_step", 0) + 1
                step = state["optimization_step"]

                state["optimization_history"].append({
                    "step": step,
                    "cycles": None,
                    "speedup": 0,
                    "correct": False,
                    "description": content[:200],
                })

                if step >= self.max_attempts:
                    state["episode_done"] = True

        return tool_messages

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    @vf.cleanup
    async def release_sandbox(self, state: State) -> None:
        sid = state.get("sandbox_id")
        if not sid:
            return

        episode_id = state.get("episode_id", sid)
        try:
            await sandbox_export_artifacts(sid, episode_id, dict(state))
        except Exception as e:
            log.warning("Failed to export artifacts for %s: %s", episode_id, e)

        self._log_episode_metrics(state)

        await self.sandbox_pool.release(sid)

    def _log_episode_metrics(self, state: State) -> None:
        """Log per-episode metrics to wandb."""
        if wandb is None or wandb.run is None:
            return

        import math

        baseline = state.get("baseline_cycles", BASELINE_CYCLES)
        best = state.get("best_cycles", baseline)
        num_steps = state.get("optimization_step", 0)
        num_failures = state.get("num_failures", 0)
        history = state.get("optimization_history", [])
        correct_non_baseline = [
            e for e in history if e.get("correct") and e.get("step", 0) > 0
        ]

        best_so_far = baseline
        num_improvements = 0
        for e in correct_non_baseline:
            if e["cycles"] is not None and e["cycles"] < best_so_far:
                num_improvements += 1
                best_so_far = e["cycles"]

        speedup = baseline / max(best, 1)
        log2_speedup = math.log2(speedup) if speedup > 1 else 0.0

        episode_data = {
            "episode/baseline_cycles": baseline,
            "episode/best_cycles": best,
            "episode/speedup": round(speedup, 2),
            "episode/log2_speedup": round(log2_speedup, 3),
            "episode/num_attempts": num_steps,
            "episode/num_failures": num_failures,
            "episode/failure_rate": num_failures / max(num_steps, 1),
            "episode/num_improvements": num_improvements,
            "episode/monotonic_frac": num_improvements / max(len(correct_non_baseline), 1),
        }
        wandb.log(episode_data)

    @vf.teardown
    async def shutdown_pool(self) -> None:
        await self.sandbox_pool.shutdown()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/robert/repos/auto-rl/kernel && python -c "import ast; ast.parse(open('kernel_opt_env/env.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add kernel/kernel_opt_env/env.py
git commit -m "feat(kernel): add KernelOptEnv environment class"
```

---

### Task 6: Entrypoint and system prompt

**Files:**
- Modify: `kernel/kernel_opt_env/__init__.py`

- [ ] **Step 1: Write full __init__.py**

Replace the placeholder with the complete entrypoint:

```python
"""
kernel-opt-env: RL environment for VLIW SIMD kernel optimization.

Exposes load_environment() as the prime-rl / verifiers entrypoint.
"""

from __future__ import annotations

from datasets import Dataset

from .env import KernelOptEnv

SYSTEM_PROMPT_TEMPLATE = """\
You are a performance engineering agent. Your task is to optimize a kernel
running on a custom VLIW SIMD machine simulator, minimizing clock cycles.

The kernel performs a batched tree traversal with hashing. The baseline
implementation is a naive scalar version running in 147,734 cycles. Your
goal is to reduce this as much as possible.

The machine has these engines executing in parallel per cycle:
  - alu:   12 scalar ALU slots
  - valu:  6 vector ALU slots (VLEN=8 elements each)
  - load:  2 load slots (scalar or vector)
  - store: 2 store slots (scalar or vector)
  - flow:  1 flow control slot

Key: all slots in one instruction bundle execute simultaneously in one cycle.
Effects (writes) don't take effect until end of cycle.

Your tools:
  read_file(path)              — read perf_takehome.py, problem.py, or test output
  edit_file(path, content)     — rewrite perf_takehome.py (full file, not a diff)
  run_tests()                  — run submission tests, get correctness + cycle count
  run_analysis(script)         — run an inline Python script to analyze bottlenecks
  run_command(command)         — run a shell command (cp, ls, cat, diff, head, tail, wc, grep)

The optimization loop:
  1. Read perf_takehome.py and problem.py to understand the current state.
  2. Use run_analysis() to profile the kernel — inspect slot utilization,
     cycle distribution, instruction counts, or any other metric.
  3. Identify the bottleneck and form an optimization hypothesis.
  4. Edit perf_takehome.py to implement it.
  5. Run tests to verify correctness and measure cycles.
  6. If correct and faster: checkpoint with run_command('cp perf_takehome.py best.py').
     If incorrect or slower: restore with run_command('cp best.py perf_takehome.py')
     and try something else.
  7. Repeat.

Do NOT modify problem.py or anything in tests/. The submission tests use
a frozen copy of the simulator.

Optimization strategies to consider:
  - VLIW instruction packing (fill multiple slots per bundle)
  - SIMD vectorization (process 8 batch elements per vector op)
  - Loop construction with jump/cond_jump instead of full unrolling
  - Instruction scheduling to maximize parallelism
  - Precomputation and constant folding

You have {max_attempts} optimization attempts. Be systematic — understand what the
bottleneck is before changing code.
"""

DATASET_REPEAT_N = 10_000


def _build_dataset(system_prompt: str) -> Dataset:
    task = {
        "task_id": "kernel-opt-baseline",
        "prompt": system_prompt,
        "answer": 1487,  # Opus 4.5 after 11.5 hours
        "info": {
            "baseline_cycles": 147734,
            "target_thresholds": [18532, 2164, 1790, 1579, 1548, 1487, 1363],
        },
    }
    return Dataset.from_list([task] * DATASET_REPEAT_N)


def load_environment(
    max_attempts: int = 20,
    sandbox_pool_size: int = 4,
    **kwargs,
) -> KernelOptEnv:
    """
    Prime-rl / verifiers entrypoint.

    Args:
        max_attempts: Maximum optimization attempts per episode.
        sandbox_pool_size: Number of sandbox directories to pre-create.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_attempts=max_attempts)
    dataset = _build_dataset(system_prompt)

    return KernelOptEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        max_attempts=max_attempts,
        sandbox_pool_size=sandbox_pool_size,
        **kwargs,
    )
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/robert/repos/auto-rl/kernel && python -c "import ast; ast.parse(open('kernel_opt_env/__init__.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add kernel/kernel_opt_env/__init__.py
git commit -m "feat(kernel): add entrypoint with system prompt and dataset"
```

---

### Task 7: Training config

**Files:**
- Create: `kernel/rl.toml`

- [ ] **Step 1: Write rl.toml**

Adapted from nanochat's rl.toml, following the same config structure (which is the actual prime-rl format, not the spec's aspirational format):

```toml
max_steps = 5000
seq_len = 32768

[model]
name = "Qwen/Qwen3-14B-Instruct"
implementation = "auto"

[model.lora]
enabled = true
rank = 128
alpha = 256
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

[optimization]
learning_rate = 5e-7
lr_scheduler = "cosine"
warmup_steps = 150
max_grad_norm = 0.5

[orchestrator]
batch_size = 64
micro_batch_size = 4
gradient_accumulation_steps = 16
rollouts_per_example = 8       # 8 rollouts per seed; cheap oracle
max_tokens = 8192              # full perf_takehome.py rewrite + reasoning
num_workers = 8                # 8 parallel rollouts (CPU-only oracle)
max_pending_rollouts = 32
rollout_timeout_secs = 600     # 10min: 20 attempts * ~3s each + overhead

[orchestrator.env]
id = "kernel-opt"

[orchestrator.env.env_args]
max_attempts = 20
sandbox_pool_size = 8

[orchestrator.algorithm]
name = "grpo"
epsilon = 0.2
epsilon_high = 0.25
use_advantage_normalization = true
use_difficulty_buffer = false  # single fixed task

[orchestrator.sampling]
temperature = 0.8
top_p = 0.95

[inference]
tensor_parallel_size = 2
max_model_len = 32768
gpu_memory_utilization = 0.85
enable_auto_tool_choice = true
tool_call_parser = "hermes"

[checkpoint]
save_on_completion = true

[logging]
wandb_project = "kernel-opt-rl"
log_every_n_steps = 5
eval_every_n_steps = 100
eval_num_examples = 8
```

- [ ] **Step 2: Commit**

```bash
git add kernel/rl.toml
git commit -m "feat(kernel): add prime-rl training config"
```

---

### Task 8: Integration test

**Files:**
- None created — this is a verification step

- [ ] **Step 1: Verify the package imports cleanly**

Run: `cd /home/robert/repos/auto-rl/kernel && pip install -e . 2>&1 | tail -5 && python -c "from kernel_opt_env import load_environment; print('import OK')"`
Expected: `import OK`

- [ ] **Step 2: Verify sandbox can be created and reset**

Run from the kernel directory:
```bash
cd /home/robert/repos/auto-rl/kernel && python -c "
import asyncio
from kernel_opt_env.sandbox import KernelOptSandboxPool, _sandbox, sandbox_run_tests

async def test():
    pool = KernelOptSandboxPool(size=1)
    await pool.initialize()

    sid = await pool.acquire()
    sb = pool._all[sid]
    print(f'Sandbox path: {sb.path}')
    print(f'Files: {sorted(f.name for f in sb.path.iterdir())}')

    # Verify perf_takehome.py exists
    assert (sb.path / 'perf_takehome.py').exists()
    assert (sb.path / 'tests' / 'submission_tests.py').exists()
    print('Files OK')

    await pool.release(sid)
    print('Release OK')
    await pool.shutdown()

asyncio.run(test())
"
```
Expected: sandbox created, files present, release succeeds.

- [ ] **Step 3: Verify run_tests works with baseline code**

Run from the kernel directory:
```bash
cd /home/robert/repos/auto-rl/kernel && python -c "
import asyncio
from kernel_opt_env.sandbox import KernelOptSandboxPool, sandbox_run_tests
from kernel_opt_env import sandbox as sandbox_mod

async def test():
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()

    sid = await pool.acquire()
    result = await sandbox_run_tests(sid)
    print(f'Crashed: {result[\"crashed\"]}')
    print(f'Correct: {result[\"correct\"]}')
    print(f'Cycles: {result.get(\"cycles\")}')
    print(f'Speedup: {result.get(\"speedup\")}')

    await pool.release(sid)
    await pool.shutdown()

asyncio.run(test())
"
```
Expected: `Correct: True`, `Cycles: 147734` (the baseline).

- [ ] **Step 4: Verify run_analysis works**

Run from the kernel directory:
```bash
cd /home/robert/repos/auto-rl/kernel && python -c "
import asyncio
from kernel_opt_env.sandbox import KernelOptSandboxPool, sandbox_run_analysis
from kernel_opt_env import sandbox as sandbox_mod

async def test():
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()
    sid = await pool.acquire()

    result = await sandbox_run_analysis(sid, 'print(\"hello from analysis\")')
    print(f'Success: {result[\"success\"]}')
    print(f'Output: {result[\"output\"].strip()}')

    await pool.release(sid)
    await pool.shutdown()

asyncio.run(test())
"
```
Expected: `Success: True`, `Output: hello from analysis`

- [ ] **Step 5: Verify run_command works**

Run from the kernel directory:
```bash
cd /home/robert/repos/auto-rl/kernel && python -c "
import asyncio
from kernel_opt_env.sandbox import KernelOptSandboxPool, sandbox_run_command
from kernel_opt_env import sandbox as sandbox_mod

async def test():
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()
    sid = await pool.acquire()

    # Test allowed command
    result = await sandbox_run_command(sid, 'ls perf_takehome.py')
    print(f'ls: {result}')

    # Test cp checkpoint
    result = await sandbox_run_command(sid, 'cp perf_takehome.py best.py')
    print(f'cp: {result}')

    # Test disallowed command
    result = await sandbox_run_command(sid, 'rm best.py')
    print(f'rm (should fail): {result}')

    await pool.release(sid)
    await pool.shutdown()

asyncio.run(test())
"
```
Expected: ls and cp succeed, rm is rejected.

- [ ] **Step 6: Commit any fixes if needed**

If any tests revealed issues, fix and commit.
