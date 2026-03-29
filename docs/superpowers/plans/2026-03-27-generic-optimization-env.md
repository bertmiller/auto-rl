# Generic Optimization Environment Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a declarative config-driven RL environment framework so new optimization problems require only a TOML file + challenge files, with Python plugin hooks for complex cases.

**Architecture:** A single `OptimizationEnv(StatefulToolEnv)` loads problem config from TOML, creates a directory-based sandbox pool, registers standard tools (read_file, edit_file, run_eval, run_command) plus optional plugin tools, and composes reward from a library of primitives. The env_response method intercepts run_eval results, parses the metric via regex, and updates episode state.

**Tech Stack:** Python 3.10+, verifiers (StatefulToolEnv, Rubric), tomli/tomllib, datasets, asyncio

---

## File Structure

```
auto_rl/
  __init__.py       # load_environment(config_path) entry point
  config.py         # ProblemConfig dataclass + TOML loading
  env.py            # OptimizationEnv(StatefulToolEnv)
  sandbox.py        # SandboxPool (dir_copy isolation)
  tools.py          # Tool factories bound to config + pool
  rewards.py        # Composable reward functions
pyproject.toml      # auto-rl package
tests/
  test_config.py
  test_sandbox.py
  test_rewards.py
  test_env.py
  fixtures/
    sample_problem/
      problem.toml
      challenge/
        solution.py
        verify.py
```

---

### Task 1: ProblemConfig + TOML loader

**Files:**
- Create: `auto_rl/config.py`
- Create: `tests/test_config.py`
- Create: `tests/fixtures/sample_problem/problem.toml`

- [ ] **Step 1: Create sample problem TOML fixture**

```toml
# tests/fixtures/sample_problem/problem.toml
[problem]
name = "sample-maximize"
system_prompt = """You are an optimization agent. Edit solution.py to maximize the score.

Your tools:
  read_file(path)        — read solution.py or verify.py
  edit_file(path, content) — rewrite solution.py (full file)
  run_eval()             — run verification, get score
  run_command(command)    — run shell commands (cp, ls, diff, cat, head, tail, wc, grep)

You have {max_attempts} attempts. Be systematic."""

[files]
challenge_dir = "challenge"
editable = ["solution.py"]
readable = ["solution.py", "verify.py"]

[oracle]
command = "python3 verify.py"
metric_pattern = 'score=([\d.]+)'
metric_name = "score"
direction = "maximize"
timeout_secs = 30

[reward]
type = "log"
baseline = 1.0

[episode]
max_attempts = 10
sandbox_pool_size = 2
```

- [ ] **Step 2: Create challenge files for sample problem**

```python
# tests/fixtures/sample_problem/challenge/solution.py
def solve():
    return 1.0

# tests/fixtures/sample_problem/challenge/verify.py
from solution import solve
result = solve()
print(f"score={result}")
```

- [ ] **Step 3: Write test for config loading**

```python
# tests/test_config.py
from pathlib import Path
from auto_rl.config import ProblemConfig, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"

def test_load_config():
    config = load_config(FIXTURE)
    assert config.name == "sample-maximize"
    assert config.metric_direction == "maximize"
    assert config.oracle_command == "python3 verify.py"
    assert "solution.py" in config.editable_files
    assert "verify.py" in config.readable_files
    assert config.reward_type == "log"
    assert config.max_attempts == 10
    # challenge_dir resolved relative to TOML location
    assert config.challenge_dir.exists()

def test_config_defaults():
    config = load_config(FIXTURE)
    assert config.sandbox_pool_size == 2
    assert config.oracle_timeout == 30

def test_config_system_prompt_template():
    config = load_config(FIXTURE)
    rendered = config.render_system_prompt(max_attempts=10, baseline_metric=1.0)
    assert "10 attempts" in rendered
```

- [ ] **Step 4: Run test, verify it fails**

Run: `cd /home/robert/repos/auto-rl && python -m pytest tests/test_config.py -v`
Expected: ImportError (auto_rl.config doesn't exist yet)

- [ ] **Step 5: Implement ProblemConfig**

```python
# auto_rl/config.py
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib


@dataclass
class ProblemConfig:
    name: str
    system_prompt: str
    challenge_dir: Path
    editable_files: list[str]
    readable_files: list[str]
    oracle_command: str
    metric_pattern: str
    metric_name: str
    metric_direction: str  # "minimize" or "maximize"
    oracle_timeout: int = 30
    reward_type: str = "log"  # "log" or "linear"
    reward_baseline: float | None = None
    reward_normalization: float = 1.0
    max_attempts: int = 20
    sandbox_pool_size: int = 4
    # Plugin hooks (dotted path "module:function")
    setup_hook: str | None = None
    reset_hook: str | None = None
    extra_tools: list[str] = field(default_factory=list)
    # Dataset
    dataset_repeat: int = 10_000
    # Raw dict for any extra fields
    extra: dict[str, Any] = field(default_factory=dict)

    def render_system_prompt(self, **kwargs) -> str:
        return self.system_prompt.format(**kwargs)

    def is_improvement(self, new: float, best: float) -> bool:
        if self.metric_direction == "minimize":
            return new < best
        return new > best

    def resolve_hook(self, hook_path: str) -> Callable:
        module_path, func_name = hook_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)


def load_config(toml_path: str | Path) -> ProblemConfig:
    toml_path = Path(toml_path)
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    prob = raw.get("problem", {})
    files = raw.get("files", {})
    oracle = raw.get("oracle", {})
    reward = raw.get("reward", {})
    episode = raw.get("episode", {})
    plugins = raw.get("plugins", {})

    # Resolve challenge_dir relative to TOML file location
    challenge_dir = toml_path.parent / files.get("challenge_dir", "challenge")

    return ProblemConfig(
        name=prob["name"],
        system_prompt=prob.get("system_prompt", ""),
        challenge_dir=challenge_dir,
        editable_files=files.get("editable", []),
        readable_files=files.get("readable", []),
        oracle_command=oracle["command"],
        metric_pattern=oracle["metric_pattern"],
        metric_name=oracle.get("metric_name", "score"),
        metric_direction=oracle.get("direction", "minimize"),
        oracle_timeout=oracle.get("timeout_secs", 30),
        reward_type=reward.get("type", "log"),
        reward_baseline=reward.get("baseline"),
        reward_normalization=reward.get("normalization", 1.0),
        max_attempts=episode.get("max_attempts", 20),
        sandbox_pool_size=episode.get("sandbox_pool_size", 4),
        setup_hook=plugins.get("setup_hook"),
        reset_hook=plugins.get("reset_hook"),
        extra_tools=plugins.get("extra_tools", []),
        dataset_repeat=episode.get("dataset_repeat", 10_000),
        extra=raw,
    )
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `cd /home/robert/repos/auto-rl && python -m pytest tests/test_config.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add auto_rl/config.py tests/test_config.py tests/fixtures/
git commit -m "feat(auto_rl): add ProblemConfig and TOML loader"
```

---

### Task 2: SandboxPool

**Files:**
- Create: `auto_rl/sandbox.py`
- Create: `tests/test_sandbox.py`

- [ ] **Step 1: Write sandbox pool tests**

```python
# tests/test_sandbox.py
import asyncio
from pathlib import Path
from auto_rl.sandbox import SandboxPool

CHALLENGE = Path(__file__).parent / "fixtures" / "sample_problem" / "challenge"

def test_pool_init_and_acquire():
    pool = SandboxPool(size=2, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    assert sid.startswith("sandbox-")
    sandbox = pool.get(sid)
    assert (sandbox.path / "solution.py").exists()
    assert (sandbox.path / "verify.py").exists()

def test_pool_release_resets():
    pool = SandboxPool(size=1, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    sandbox = pool.get(sid)
    (sandbox.path / "solution.py").write_text("modified")
    (sandbox.path / "junk.py").write_text("junk")
    asyncio.run(pool.release(sid))
    sid2 = asyncio.run(pool.acquire())
    sandbox2 = pool.get(sid2)
    assert sandbox2.path == sandbox.path
    assert "modified" not in (sandbox2.path / "solution.py").read_text()
    assert not (sandbox2.path / "junk.py").exists()

def test_exec():
    from auto_rl.sandbox import sandbox_exec
    pool = SandboxPool(size=1, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    code, stdout, stderr = asyncio.run(sandbox_exec(pool, sid, "echo hello"))
    assert code == 0
    assert "hello" in stdout
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement SandboxPool**

```python
# auto_rl/sandbox.py
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Sandbox:
    id: str
    path: Path


class SandboxPool:
    def __init__(
        self,
        size: int,
        challenge_dir: Path,
        sandboxes_dir: Path | None = None,
    ):
        self.size = size
        self.challenge_dir = Path(challenge_dir)
        self.sandboxes_dir = Path(sandboxes_dir or f"/tmp/auto-rl-{uuid.uuid4().hex[:8]}")
        self._available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self._all: dict[str, Sandbox] = {}
        self._initialized = False

    def _copy_challenge(self, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.challenge_dir, dest)

    def _reset_sandbox(self, sandbox: Sandbox) -> None:
        # Remove everything and re-copy
        if sandbox.path.exists():
            shutil.rmtree(sandbox.path)
        self._copy_challenge(sandbox.path)

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.sandboxes_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        for i in range(self.size):
            sid = f"sandbox-{i}"
            path = self.sandboxes_dir / sid
            await loop.run_in_executor(None, self._copy_challenge, path)
            sandbox = Sandbox(id=sid, path=path)
            self._all[sid] = sandbox
            await self._available.put(sandbox)
        self._initialized = True
        log.info("SandboxPool: %d sandboxes from %s", self.size, self.challenge_dir)

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
        await loop.run_in_executor(None, self._reset_sandbox, sandbox)
        await self._available.put(sandbox)

    def get(self, sandbox_id: str) -> Sandbox:
        return self._all[sandbox_id]

    async def shutdown(self) -> None:
        log.info("SandboxPool shut down")


async def sandbox_exec(
    pool: SandboxPool,
    sandbox_id: str,
    cmd: str,
    timeout: int = 60,
    env_extra: dict | None = None,
) -> tuple[int, str, str]:
    import os
    sandbox = pool.get(sandbox_id)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                cwd=str(sandbox.path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 1, "", f"Timed out after {timeout}s"
    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    err = stderr.decode("utf-8", errors="replace") if stderr else ""
    return proc.returncode or 0, out, err
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add auto_rl/sandbox.py tests/test_sandbox.py
git commit -m "feat(auto_rl): add generic SandboxPool with dir_copy isolation"
```

---

### Task 3: Tool factories

**Files:**
- Create: `auto_rl/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write tool tests**

```python
# tests/test_tools.py
import asyncio
from pathlib import Path
from auto_rl.config import load_config
from auto_rl.sandbox import SandboxPool
from auto_rl.tools import create_tools

TOML = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"

def _setup():
    config = load_config(TOML)
    pool = SandboxPool(size=1, challenge_dir=config.challenge_dir)
    sid = asyncio.run(pool.acquire())
    tools = create_tools(config, pool)
    return config, pool, sid, tools

def test_read_file_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert "def solve" in result

def test_read_file_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["read_file"]("secret.py", sandbox_id=sid))
    assert "Error" in result

def test_edit_file_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["edit_file"]("solution.py", "x = 1", sandbox_id=sid))
    assert result == "OK"
    content = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert content == "x = 1"

def test_edit_file_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["edit_file"]("verify.py", "hack", sandbox_id=sid))
    assert "Error" in result

def test_run_eval():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_eval"](sandbox_id=sid))
    assert "score=" in result

def test_run_command_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_command"]("ls", sandbox_id=sid))
    assert "solution.py" in result

def test_run_command_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_command"]("rm -rf /", sandbox_id=sid))
    assert "Error" in result
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement tool factories**

```python
# auto_rl/tools.py
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
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add auto_rl/tools.py tests/test_tools.py
git commit -m "feat(auto_rl): add config-bound tool factories"
```

---

### Task 4: Reward function library

**Files:**
- Create: `auto_rl/rewards.py`
- Create: `tests/test_rewards.py`

- [ ] **Step 1: Write reward tests**

```python
# tests/test_rewards.py
import asyncio
from auto_rl.rewards import (
    log_improvement_reward,
    linear_improvement_reward,
    monotonic_improvement_reward,
    failure_penalty,
    build_rubric,
)

def _state(baseline, best, direction="minimize", failures=0, history=None):
    return {
        "baseline_metric": baseline,
        "best_metric": best,
        "metric_direction": direction,
        "reward_normalization": 1.0,
        "num_failures": failures,
        "optimization_history": history or [],
    }

def test_log_minimize_2x():
    # 2x improvement on minimize: log2(100/50) = 1.0
    s = _state(100.0, 50.0, "minimize")
    r = asyncio.run(log_improvement_reward([], s))
    assert abs(r - 1.0) < 0.01

def test_log_maximize_2x():
    s = _state(50.0, 100.0, "maximize")
    r = asyncio.run(log_improvement_reward([], s))
    assert abs(r - 1.0) < 0.01

def test_log_no_improvement():
    s = _state(100.0, 100.0, "minimize")
    r = asyncio.run(log_improvement_reward([], s))
    assert r == 0.0

def test_linear_improvement():
    s = _state(100.0, 80.0, "minimize")
    s["reward_normalization"] = 50.0
    r = asyncio.run(linear_improvement_reward([], s))
    assert abs(r - 0.4) < 0.01  # (100-80)/50

def test_failure_penalty():
    s = _state(100.0, 50.0, failures=3)
    r = asyncio.run(failure_penalty([], s))
    assert abs(r - (-0.15)) < 0.01  # -0.05 * 3

def test_monotonic():
    history = [
        {"step": 0, "correct": True, "metric": 100},
        {"step": 1, "correct": True, "metric": 90},
        {"step": 2, "correct": True, "metric": 95},  # not monotonic
        {"step": 3, "correct": True, "metric": 80},
    ]
    s = _state(100.0, 80.0, "minimize", history=history)
    r = asyncio.run(monotonic_improvement_reward([], s))
    # Steps 1,2,3 are non-baseline. Step 1: 90<100 (improve). Step 2: 95>90 (no). Step 3: 80<90 (improve).
    # 2 improvements / 3 steps = 0.667
    assert abs(r - 2 / 3) < 0.01

def test_build_rubric():
    rubric = build_rubric("log", weights={"improvement": 1.0, "monotonic": 0.25})
    assert rubric is not None
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement reward functions**

```python
# auto_rl/rewards.py
from __future__ import annotations

import math

import verifiers as vf


async def log_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    baseline = state["baseline_metric"]
    best = state["best_metric"]
    direction = state["metric_direction"]

    if direction == "minimize":
        if best >= baseline:
            return 0.0
        ratio = baseline / best
    else:
        if best <= baseline:
            return 0.0
        ratio = best / baseline

    return float(min(math.log2(ratio), 10.0))


async def linear_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    baseline = state["baseline_metric"]
    best = state["best_metric"]
    direction = state["metric_direction"]
    normalization = state.get("reward_normalization", 1.0)

    if direction == "minimize":
        delta = baseline - best
    else:
        delta = best - baseline

    return float(max(0.0, min(delta / normalization, 10.0)))


async def monotonic_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    history = state.get("optimization_history", [])
    direction = state.get("metric_direction", "minimize")
    non_baseline = [e for e in history if e.get("correct") and e.get("step", 0) > 0 and e.get("metric") is not None]
    if not non_baseline:
        return 0.0

    improvements = 0
    best_so_far = state["baseline_metric"]
    for e in non_baseline:
        m = e["metric"]
        if (direction == "minimize" and m < best_so_far) or \
           (direction == "maximize" and m > best_so_far):
            improvements += 1
            best_so_far = m

    return improvements / len(non_baseline)


async def failure_penalty(completion: list, state: dict, **kwargs) -> float:
    return -0.05 * state.get("num_failures", 0)


def build_rubric(
    reward_type: str = "log",
    weights: dict[str, float] | None = None,
) -> vf.Rubric:
    weights = weights or {}
    improvement_fn = log_improvement_reward if reward_type == "log" else linear_improvement_reward
    funcs = [improvement_fn, monotonic_improvement_reward, failure_penalty]
    w = [
        weights.get("improvement", 1.0),
        weights.get("monotonic", 0.25),
        weights.get("failure", 1.0),
    ]
    return vf.Rubric(funcs=funcs, weights=w)
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add auto_rl/rewards.py tests/test_rewards.py
git commit -m "feat(auto_rl): add composable reward function library"
```

---

### Task 5: OptimizationEnv

**Files:**
- Create: `auto_rl/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write env tests**

```python
# tests/test_env.py
import asyncio
from pathlib import Path
from auto_rl.config import load_config
from auto_rl.env import OptimizationEnv

TOML = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"

def test_env_creates():
    config = load_config(TOML)
    env = OptimizationEnv(config)
    assert env.config.name == "sample-maximize"
    assert len(env.tool_defs) == 4  # read_file, edit_file, run_eval, run_command

def test_env_setup_state():
    config = load_config(TOML)
    env = OptimizationEnv(config)
    state = {
        "info": {"baseline_metric": 1.0},
        "task": "sample",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    assert "sandbox_id" in state
    assert state["baseline_metric"] == 1.0
    assert state["best_metric"] == 1.0
    assert state["optimization_step"] == 0
    assert state["episode_done"] is False
    # Cleanup
    asyncio.run(env.sandbox_pool.release(state["sandbox_id"]))
```

- [ ] **Step 2: Run tests, verify they fail**

- [ ] **Step 3: Implement OptimizationEnv**

```python
# auto_rl/env.py
from __future__ import annotations

import logging
import math
import re
import uuid

import verifiers as vf
from verifiers.types import Messages, State

from .config import ProblemConfig
from .rewards import build_rubric
from .sandbox import SandboxPool
from .tools import create_tools

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class OptimizationEnv(vf.StatefulToolEnv):
    """
    Generic RL environment for optimization-by-code-editing.

    Loads behavior from a ProblemConfig. Handles sandbox lifecycle,
    tool registration, metric tracking, and reward computation.
    """

    def __init__(self, config: ProblemConfig, **kwargs):
        self.config = config
        rubric = build_rubric(
            reward_type=config.reward_type,
            weights=kwargs.pop("reward_weights", None),
        )
        super().__init__(
            rubric=rubric,
            max_turns=config.max_attempts * 4,
            **kwargs,
        )
        self.sandbox_pool = SandboxPool(
            size=config.sandbox_pool_size,
            challenge_dir=config.challenge_dir,
        )
        self._tools = create_tools(config, self.sandbox_pool)

        for tool_fn in self._tools.values():
            self.add_tool(tool_fn, args_to_skip=["sandbox_id"])

        # Register plugin tools
        for hook_path in config.extra_tools:
            fn = config.resolve_hook(hook_path)
            self.add_tool(fn, args_to_skip=["sandbox_id"])

    async def setup_state(self, state: State, **kwargs) -> State:
        sandbox_id = await self.sandbox_pool.acquire()

        task_info = state.get("info", {})
        baseline = task_info.get("baseline_metric", self.config.reward_baseline or 0.0)

        # Run setup hook if configured
        if self.config.setup_hook:
            hook = self.config.resolve_hook(self.config.setup_hook)
            await hook(sandbox_id=sandbox_id, state=state, pool=self.sandbox_pool, config=self.config)

        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        state.update({
            "sandbox_id": sandbox_id,
            "episode_id": episode_id,
            "baseline_metric": baseline,
            "best_metric": baseline,
            "current_metric": baseline,
            "metric_direction": self.config.metric_direction,
            "metric_name": self.config.metric_name,
            "reward_normalization": self.config.reward_normalization,
            "optimization_history": [{
                "step": 0,
                "metric": baseline,
                "correct": True,
                "description": "baseline",
            }],
            "num_failures": 0,
            "optimization_step": 0,
            "episode_done": False,
        })
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages, state: State, **kwargs
    ) -> dict:
        updated = dict(tool_args)
        updated["sandbox_id"] = state["sandbox_id"]
        return updated

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        # Find run_eval call IDs
        eval_call_ids: set[str] = set()
        last_msg = messages[-1] if messages else {}
        tool_calls = (
            last_msg.get("tool_calls", [])
            if isinstance(last_msg, dict)
            else getattr(last_msg, "tool_calls", []) or []
        )
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            if name == "run_eval":
                eval_call_ids.add(tc_id)

        tool_messages = await super().env_response(messages, state, **kwargs)

        for msg in tool_messages:
            tc_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
            if tc_id not in eval_call_ids:
                continue

            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if not content:
                continue

            metric_key = f"{self.config.metric_name}="
            if metric_key in content and "CRASH" not in content:
                m = re.search(rf"{re.escape(self.config.metric_name)}=([\d.eE+\-]+)", content)
                if m:
                    metric_val = float(m.group(1))
                    state["optimization_step"] = state.get("optimization_step", 0) + 1
                    step = state["optimization_step"]
                    state["current_metric"] = metric_val

                    if self.config.is_improvement(metric_val, state["best_metric"]):
                        state["best_metric"] = metric_val

                    state["optimization_history"].append({
                        "step": step,
                        "metric": metric_val,
                        "correct": True,
                        "description": "",
                    })
                    if step >= self.config.max_attempts:
                        state["episode_done"] = True

            elif "CRASH" in content or "ERROR" in content:
                state["num_failures"] = state.get("num_failures", 0) + 1
                state["optimization_step"] = state.get("optimization_step", 0) + 1
                step = state["optimization_step"]
                state["optimization_history"].append({
                    "step": step,
                    "metric": None,
                    "correct": False,
                    "description": content[:200],
                })
                if step >= self.config.max_attempts:
                    state["episode_done"] = True

        return tool_messages

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    async def rollout(self, *args, **kwargs):
        state = await super().rollout(*args, **kwargs)
        await self._release_sandbox(state)
        return state

    async def _release_sandbox(self, state: State) -> None:
        sid = state.get("sandbox_id")
        if not sid:
            return
        self._log_episode_metrics(state)
        await self.sandbox_pool.release(sid)

    def _log_episode_metrics(self, state: State) -> None:
        if wandb is None or wandb.run is None:
            return
        baseline = state.get("baseline_metric", 0)
        best = state.get("best_metric", baseline)
        steps = state.get("optimization_step", 0)
        failures = state.get("num_failures", 0)
        wandb.log({
            "episode/baseline": baseline,
            "episode/best": best,
            "episode/steps": steps,
            "episode/failures": failures,
            "episode/failure_rate": failures / max(steps, 1),
        })

    @vf.teardown
    async def shutdown_pool(self) -> None:
        await self.sandbox_pool.shutdown()
```

- [ ] **Step 4: Run tests, verify they pass**

- [ ] **Step 5: Commit**

```bash
git add auto_rl/env.py tests/test_env.py
git commit -m "feat(auto_rl): add generic OptimizationEnv"
```

---

### Task 6: Entry point and packaging

**Files:**
- Create: `auto_rl/__init__.py`
- Create: `pyproject.toml` (root, for auto-rl package)

- [ ] **Step 1: Create package init with load_environment**

```python
# auto_rl/__init__.py
"""
auto-rl: Generic RL environments for optimization-by-code-editing.

Usage:
    from auto_rl import load_environment
    env = load_environment("path/to/problem.toml")
"""

from .config import ProblemConfig, load_config
from .env import OptimizationEnv
from .rewards import build_rubric
from .sandbox import SandboxPool

from datasets import Dataset


def load_environment(config_path: str, **overrides) -> OptimizationEnv:
    """
    Load an optimization RL environment from a TOML config file.

    Args:
        config_path: Path to problem.toml
        **overrides: Override config fields (max_attempts, sandbox_pool_size, etc.)

    Returns:
        Configured OptimizationEnv ready for training.
    """
    config = load_config(config_path)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Build dataset
    baseline = config.reward_baseline or 0.0
    system_prompt = config.render_system_prompt(
        max_attempts=config.max_attempts,
        metric_name=config.metric_name,
        baseline_metric=baseline,
    )
    task = {
        "task_id": f"{config.name}-baseline",
        "prompt": system_prompt,
        "answer": str(baseline),
        "info": {"baseline_metric": baseline},
    }
    dataset = Dataset.from_list([task] * config.dataset_repeat)

    return OptimizationEnv(
        config=config,
        dataset=dataset,
        system_prompt=system_prompt,
    )
```

- [ ] **Step 2: Create pyproject.toml**

```toml
# pyproject.toml (root)
[project]
name = "auto-rl"
version = "0.1.0"
description = "Generic RL environments for optimization-by-code-editing"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.0",
    "datasets",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["auto_rl*"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Run full test suite**

Run: `cd /home/robert/repos/auto-rl && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add auto_rl/__init__.py pyproject.toml
git commit -m "feat(auto_rl): add load_environment entry point and packaging"
```

---

### Task 7: End-to-end validation with sample problem

**Files:**
- Modify: `tests/test_env.py` (add integration test)

- [ ] **Step 1: Add integration test that exercises the full loop**

```python
# Append to tests/test_env.py

def test_env_full_tool_loop():
    """Simulate an agent editing solution.py and running eval."""
    config = load_config(TOML)
    env = OptimizationEnv(config)
    state = {"info": {"baseline_metric": 1.0}, "task": "sample", "answer": ""}
    state = asyncio.run(env.setup_state(state))

    sid = state["sandbox_id"]
    tools = env._tools

    # Read the solution
    result = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert "def solve" in result

    # Edit to return a higher score
    new_code = "def solve():\n    return 42.0\n"
    result = asyncio.run(tools["edit_file"]("solution.py", new_code, sandbox_id=sid))
    assert result == "OK"

    # Run eval
    result = asyncio.run(tools["run_eval"](sandbox_id=sid))
    assert "score=42.0" in result

    # Checkpoint
    result = asyncio.run(tools["run_command"]("cp solution.py best.py", sandbox_id=sid))

    # Cleanup
    asyncio.run(env.sandbox_pool.release(sid))
```

- [ ] **Step 2: Run, verify it passes**

Run: `cd /home/robert/repos/auto-rl && python -m pytest tests/test_env.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_env.py
git commit -m "test(auto_rl): add end-to-end integration test"
```
