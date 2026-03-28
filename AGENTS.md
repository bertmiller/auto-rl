# auto-rl

RL environments for training LLM agents to optimize code through iterative edit-test loops.
Built on [verifiers](https://github.com/willccbb/verifiers) + [prime-rl](https://github.com/PRIME-RL/PRIME-RL).

## What this is

A collection of RL environments where an agent:
1. Reads code and reference material
2. Edits a solution file
3. Runs a deterministic oracle to measure a scalar metric
4. Gets reward proportional to improvement

Each environment is a self-contained Python package exposing `load_environment()` as
a `verifiers.envs` entry point. Training uses GRPO via prime-rl.

## Repository structure

```
auto-rl/
├── kernel/                     # VLIW SIMD kernel optimization (CPU-only oracle)
│   ├── kernel_opt_env/         # Environment package
│   │   ├── __init__.py         # Entry point, dataset, system prompt, starting points
│   │   ├── env.py              # KernelOptEnv (StatefulToolEnv subclass)
│   │   ├── sandbox.py          # Directory-based sandbox pool
│   │   ├── tools.py            # read_file, edit_file, run_tests, run_analysis, run_command
│   │   └── rubric.py           # log2 speedup + monotonic improvement + auxiliary rewards
│   ├── original_performance_takehome/  # Challenge files (read-only reference)
│   ├── starting_points/        # 8 pre-optimized variants for curriculum
│   ├── test_episode.py         # Manual episode runner (OpenAI-compatible API)
│   ├── spec.md                 # Full MDP spec
│   ├── rl.toml                 # prime-rl training config
│   └── pyproject.toml
│
├── nanochat/                   # karpathy/autoresearch optimization (GPU oracle)
│   ├── autoresearch_env/       # Environment package
│   │   ├── __init__.py         # Entry point, dataset, system prompt
│   │   ├── env.py              # AutoresearchEnv (StatefulToolEnv subclass)
│   │   ├── sandbox.py          # Git-branch sandbox pool with GPU pinning
│   │   ├── tools.py            # read_file, edit_file, run_experiment, git_commit_or_revert
│   │   └── rubric.py           # val_bpb improvement + monotonic improvement
│   ├── spec.md
│   ├── rl.toml
│   └── pyproject.toml
│
└── docs/
    └── ideas.md                # Future environment ideas
```

## Environment anatomy

Every environment follows the same 5-module pattern:

| Module | Purpose |
|---|---|
| `__init__.py` | `load_environment()` entry point, system prompt template, dataset construction |
| `env.py` | `StatefulToolEnv` subclass — state machine, tool arg injection, response interception |
| `sandbox.py` | Isolation pool — one sandbox per concurrent rollout, reset between episodes |
| `tools.py` | Async tool functions with `sandbox_id` injected and hidden from model schema |
| `rubric.py` | `Rubric(funcs=[...], weights=[...])` composing multiple reward signals |

## Key frameworks

- **verifiers** — provides `StatefulToolEnv`, `Rubric`, tool registration, `State`/`Messages` types
- **prime-rl** — GRPO training loop, reads `rl.toml` for config
- **datasets** (HuggingFace) — task sampling

## Environment details

### kernel/ — VLIW SIMD kernel optimization

- **Task:** Optimize `perf_takehome.py` to minimize clock cycles on a simulated VLIW SIMD machine
- **Metric:** Clock cycles (lower is better), reward is `log2(baseline / best)`
- **Oracle:** Deterministic Python simulator, ~1-5s per run, CPU-only
- **Action:** Full-file rewrite of `perf_takehome.py`
- **Tools:** `read_file`, `edit_file`, `run_tests`, `run_analysis`, `run_command`
- **Curriculum:** 8 starting points from 147,734 cycles (naive) to 2,432 cycles (60x optimized)
- **Sandbox:** Directory copies, file-level reset
- **Range:** Baseline 147,734 → best known ~1,363 cycles (108x speedup)

### nanochat/ — autoresearch optimization

- **Task:** Optimize `train.py` to minimize `val_bpb` (validation bits per byte)
- **Metric:** val_bpb (lower is better), reward is `(baseline - best) / 0.02`
- **Oracle:** 5-minute GPU training run
- **Action:** Full-file rewrite of `train.py`
- **Tools:** `read_file`, `edit_file`, `run_experiment`, `git_commit_or_revert`
- **Sandbox:** Git branch isolation with `CUDA_VISIBLE_DEVICES` pinning
- **Range:** ~0.02 val_bpb improvement is strong (Karpathy's level from ~90 experiments)

## Reward design patterns

All environments combine:
1. **Primary reward** — log-scaled or normalized improvement over baseline
2. **Process reward** — monotonic improvement fraction (rewards systematic exploration)
3. **Auxiliary rewards** — small bonuses for good practices (notes, analysis)
4. **Failure penalty** — linear penalty per incorrect/crashed attempt

## Adding a new environment

1. Create a directory: `my_env/`
2. Write the 5 modules following the pattern above
3. Add a `pyproject.toml` with a `verifiers.envs` entry point
4. Add a `rl.toml` for training config
5. Write a `spec.md` documenting the MDP
6. See `docs/ideas.md` for proposed environments

## Training

```bash
# Install environment
cd kernel && pip install -e .

# Run manual episode (requires vLLM server)
python test_episode.py

# Train with prime-rl
prime-rl train --config rl.toml
```
