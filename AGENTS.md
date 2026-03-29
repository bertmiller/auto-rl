# auto-rl

RL environments for training LLM agents to optimize code through iterative edit-test loops.
Built on [verifiers](https://github.com/willccbb/verifiers) + [prime-rl](https://github.com/PRIME-RL/PRIME-RL).

## What this is

A collection of RL environments where an agent:
1. Reads code and reference material
2. Edits a solution file
3. Runs a deterministic oracle to measure a scalar metric
4. Gets reward proportional to improvement

New environments can be created with just a TOML config file + challenge files using the
`auto_rl` framework. Complex environments can use Python plugin hooks for custom behavior.

## Repository structure

```
auto-rl/
├── auto_rl/                    # Generic RL environment framework
│   ├── __init__.py             # load_environment(config_path) entry point
│   ├── config.py               # ProblemConfig dataclass + TOML loader
│   ├── env.py                  # OptimizationEnv(StatefulToolEnv) — generic environment
│   ├── sandbox.py              # SandboxPool — dir_copy isolation, acquire/release/reset
│   ├── tools.py                # Tool factories: read_file, edit_file, run_eval, run_command
│   └── rewards.py              # Reward library: log/linear improvement, monotonic, failure penalty
│
├── problems/                   # Optimization problem environments
│   ├── kernel/                 # VLIW SIMD kernel optimization (CPU-only oracle)
│   │   ├── kernel_opt_env/     # Custom environment package (pre-dates generic framework)
│   │   ├── original_performance_takehome/  # Challenge files
│   │   ├── starting_points/    # 8 pre-optimized variants for curriculum
│   │   ├── spec.md             # Full MDP spec
│   │   ├── rl.toml             # prime-rl training config
│   │   └── test_episode.py     # Manual episode runner
│   │
│   ├── nanochat/               # karpathy/autoresearch optimization (GPU oracle)
│   │   ├── autoresearch_env/   # Custom environment package
│   │   ├── spec.md
│   │   └── rl.toml
│   │
│   └── circle-packing/         # Circle packing optimization (WIP)
│
├── tests/                      # Framework tests
│   ├── fixtures/               # Sample problem for testing
│   ├── test_config.py
│   ├── test_env.py
│   ├── test_rewards.py
│   ├── test_sandbox.py
│   └── test_tools.py
│
├── alphaevolve/                # AlphaEvolve problem reference (67 math problems)
│
├── docs/
│   ├── ideas.md                # Proposed environments + prior art survey
│   └── creating-an-environment.md  # Guide to creating new environments
│
└── pyproject.toml              # auto-rl package
```

## Generic framework (auto_rl/)

Most problems need only a TOML config + challenge files:

```toml
# problems/my-problem/problem.toml
[problem]
name = "my-problem"
system_prompt = "..."

[files]
challenge_dir = "challenge"
editable = ["solution.py"]
readable = ["solution.py", "verify.py"]

[oracle]
command = "python3 verify.py"
metric_pattern = 'score=([\d.]+)'
direction = "maximize"

[reward]
type = "log"
baseline = 1.0

[episode]
max_attempts = 20
```

```python
from auto_rl import load_environment
env = load_environment("problems/my-problem/problem.toml")
```

For complex problems, use plugin hooks:
```toml
[plugins]
setup_hook = "my_hooks:install_variant"
extra_tools = ["my_hooks:run_analysis"]
```

See `docs/creating-an-environment.md` for the full guide.

## Problem details

### problems/kernel/ — VLIW SIMD kernel optimization

- **Task:** Optimize `perf_takehome.py` to minimize clock cycles on a simulated VLIW SIMD machine
- **Metric:** Clock cycles (lower is better), reward is `log2(baseline / best)`
- **Oracle:** Deterministic Python simulator, ~1-5s per run, CPU-only
- **Action:** Full-file rewrite of `perf_takehome.py`
- **Tools:** `read_file`, `edit_file`, `run_tests`, `run_analysis`, `run_command`
- **Curriculum:** 8 starting points from 147,734 cycles (naive) to 2,432 cycles (60x optimized)
- **Range:** Baseline 147,734 → best known ~1,363 cycles (108x speedup)

### problems/nanochat/ — autoresearch optimization

- **Task:** Optimize `train.py` to minimize `val_bpb` (validation bits per byte)
- **Metric:** val_bpb (lower is better), reward is `(baseline - best) / 0.02`
- **Oracle:** 5-minute GPU training run
- **Action:** Full-file rewrite of `train.py`
- **Tools:** `read_file`, `edit_file`, `run_experiment`, `git_commit_or_revert`
- **Sandbox:** Git branch isolation with `CUDA_VISIBLE_DEVICES` pinning
- **Range:** ~0.02 val_bpb improvement is strong (Karpathy's level from ~90 experiments)

## Reward design

All environments combine:
1. **Primary reward** — log-scaled or normalized improvement over baseline
2. **Process reward** — monotonic improvement fraction (rewards systematic exploration)
3. **Failure penalty** — linear penalty per incorrect/crashed attempt

## Training

```bash
# Install framework
pip install -e .

# Run manual episode test
python test_run.py

# Train with prime-rl
prime-rl train --config problems/kernel/rl.toml
```
