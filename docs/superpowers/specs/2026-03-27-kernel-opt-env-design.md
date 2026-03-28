# kernel-opt-env Implementation Design

## Overview

Implement the RL environment described in `kernel/spec.md`: train models to optimize kernels on a custom VLIW SIMD architecture. Mirrors the nanochat `autoresearch_env` structure and patterns, adapted for the kernel optimization challenge.

## Directory Layout

```
kernel/
├── spec.md                          # (exists)
├── original_performance_takehome/   # (exists, read-only challenge source)
├── pyproject.toml                   # package def + verifiers entrypoint
├── rl.toml                          # prime-rl training config
├── .gitignore
└── kernel_opt_env/
    ├── __init__.py                  # load_environment + dataset builder
    ├── env.py                       # KernelOptEnv(StatefulToolEnv)
    ├── sandbox.py                   # KernelOptSandboxPool
    ├── tools.py                     # 5 tools
    └── rubric.py                    # reward functions
```

## Components

### sandbox.py — KernelOptSandboxPool

Pool of pre-created directory copies of `original_performance_takehome/`.

**State per sandbox:**
- `sandbox_id: str` — unique identifier
- `sandbox_dir: str` — absolute path to the working directory
- `in_use: bool`

**Lifecycle:**
- Pool init: create N sandbox directories by copying `original_performance_takehome/` contents
- `acquire() -> str`: return an available sandbox_id, mark in-use
- `release(sandbox_id)`: reset sandbox to clean state, mark available
- Reset = overwrite `perf_takehome.py` with the original baseline, remove any extra files created by the agent (e.g. `best.py`, `analysis_*.py`, `test_output.log`)

**Subprocess helpers:**
- `sandbox_run_tests(sandbox_id) -> dict`: run `python tests/submission_tests.py` in sandbox dir, parse stdout for cycle count and correctness, capture stderr for crash info. Timeout: 120s (the submission tests run 8 correctness trials).
- `sandbox_run_analysis(sandbox_id, script, timeout_secs=30) -> dict`: write script to temp file in sandbox, run `python <tempfile>`, capture stdout/stderr, clean up temp file.
- `sandbox_run_command(sandbox_id, command, timeout_secs=10) -> dict`: validate command against allowlist, run in sandbox dir, capture stdout/stderr.

**Environment variables:**
- `KERNEL_OPT_CHALLENGE_DIR`: path to `original_performance_takehome/` (default: auto-detect relative to package)
- `KERNEL_OPT_SANDBOXES_DIR`: where to create sandbox directories (default: `/tmp/kernel-opt-sandboxes`)
- `KERNEL_OPT_ARTIFACTS_DIR`: where to export episode artifacts (default: `./artifacts`)

**Key difference from nanochat:** No GPU pinning, no `uv sync`, no git branch management. Just directory copies with file reset.

### tools.py — Five Tools

All tools are async functions. `sandbox_id` is injected via `args_to_skip`.

**1. `read_file(path, sandbox_id)`**
- Allowlist: `perf_takehome.py`, `problem.py`, `test_output.log`, `tests/submission_tests.py`, `best.py`
- Note: `best.py` is on the allowlist because the agent creates it via `cp` for checkpointing
- Returns file content or error message

**2. `edit_file(path, content, sandbox_id)`**
- Only `perf_takehome.py` may be written
- Full-file overwrite (not a diff)
- Returns `"OK"` or error

**3. `run_tests(sandbox_id)`**
- Calls `sandbox_run_tests`
- Parses output to extract: cycle count, correctness (pass/fail), number of speed tests passed
- Returns formatted string: `"CORRECT | cycles=18532 | speedup=7.97x | tests_passed=3/8"` or `"INCORRECT | ..."` or `"CRASH | ..."`

**4. `run_analysis(script, sandbox_id)`**
- Calls `sandbox_run_analysis`
- 30-second timeout
- Stdout truncated to 10,000 chars
- Returns stdout or error/timeout message

**5. `run_command(command, sandbox_id)`**
- Allowlist of command prefixes: `cp`, `ls`, `cat`, `diff`, `head`, `tail`, `wc`, `grep`
- Calls `sandbox_run_command`
- 10-second timeout
- Returns stdout/stderr or error
- Rejects commands not starting with an allowed prefix

### env.py — KernelOptEnv

Subclass of `vf.StatefulToolEnv`.

**Per-rollout state:**
```python
{
    "sandbox_id": str,
    "baseline_cycles": 147734,          # constant
    "best_cycles": 147734,              # updated on successful runs
    "current_cycles": 147734,           # most recent test result
    "optimization_history": [           # full history
        {"step": 0, "cycles": 147734, "speedup": 1.0, "correct": True, "description": "baseline"}
    ],
    "num_failures": 0,                  # incorrect results or crashes
    "episode_done": False,
}
```

**Methods:**
- `setup_state`: acquire sandbox, initialize state with baseline constants (no run needed — deterministic baseline)
- `update_tool_args`: inject `sandbox_id` into all tool calls
- `env_response`: intercept `run_tests` tool results to update `best_cycles`, `current_cycles`, `optimization_history`, `num_failures`
- `is_completed`: delegate to super (max_turns) or check `episode_done`
- `release_sandbox` (cleanup): export artifacts, release sandbox back to pool

**Artifact export on cleanup:**
- `perf_takehome.py` (final version)
- `best.py` (if it exists — agent's checkpointed best)
- `optimization_history.json` (full state history)

### rubric.py — Reward Functions

**1. `speedup_reward` (weight: 1.0)**
```
log2(147734 / best_cycles) - 0.05 * num_failures
```
Clamped to [0.0, 10.0].

Reference: 2x speedup = 1.0, 8x = 3.0, 100x = 6.6.

**2. `monotonic_improvement_reward` (weight: 0.25)**

Fraction of correct, non-baseline attempts that set a new best cycle count. Rewards systematic hill-climbing over random thrashing.

### __init__.py — Entry Point

**`load_environment(config) -> KernelOptEnv`**: factory function registered as verifiers entrypoint.

**Dataset:** single repeated task (10,000 copies). Each entry contains:
- `task_id`: `"kernel-opt-baseline"`
- `prompt`: system prompt from spec (describes the machine, tools, optimization loop)
- `answer`: `1487` (Opus 4.5 11.5hr reference)
- `info`: baseline_cycles, target_thresholds

### pyproject.toml

```toml
[project]
name = "kernel-opt-env"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["verifiers>=0.1.0", "datasets"]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[project.entry-points."verifiers.envs"]
kernel-opt = "kernel_opt_env:load_environment"
```

### rl.toml

Pulled directly from spec section 8, adjusted to match nanochat's proven config structure. Key differences from nanochat:
- 64 workers (cheap oracle)
- 8 rollouts per seed
- 600s rollout timeout (vs 5400s)
- `max_attempts = 20` (vs `max_experiments = 15`)

## Testing Strategy

- Unit test sandbox pool: create, acquire, release, reset
- Unit test each tool: valid inputs, invalid inputs, allowlist enforcement
- Integration test: full episode with baseline code (should get 147734 cycles)
- Integration test: manually optimized code to verify reward computation
- Run `python tests/submission_tests.py` in a sandbox to verify the frozen simulator works in isolation

## Decisions

1. **Sandbox isolation:** directory copies with file-level reset (no containers, no git branches)
2. **Version management:** agent uses `run_command("cp ...")` to checkpoint/restore — learns version management as part of optimization strategy
3. **Command allowlist:** `cp`, `ls`, `cat`, `diff`, `head`, `tail`, `wc`, `grep` — sufficient for file management and inspection
4. **No baseline run at episode start:** cycle count is a known constant (147,734) since the simulator is deterministic
5. **Separate package:** own `pyproject.toml` with verifiers entrypoint, independent of nanochat
