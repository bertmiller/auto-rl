# Creating a New Environment

This guide walks through creating a new optimization environment using the `auto_rl` framework.

## Quick start

A new environment needs two things:

1. **A challenge directory** — the files the agent works with
2. **A `problem.toml`** — config that tells the framework how to run it

```
my_problem/
  problem.toml
  challenge/
    solution.py      # the file the agent edits
    verify.py        # the oracle that scores it
    reference.py     # any read-only reference material
```

Then load it:

```python
from auto_rl import load_environment
env = load_environment("my_problem/problem.toml")
```

## Step 1: Write the oracle

The oracle is a script that runs the agent's solution and prints a metric. It must:

- Exit 0 on success, non-zero on crash
- Print the metric somewhere in stdout in a parseable format (e.g. `score=42.5`)

```python
# challenge/verify.py
from solution import solve

result = solve()
valid = check_constraints(result)
if not valid:
    print("INVALID: constraint violated")
    exit(1)
print(f"score={compute_score(result)}")
```

The framework parses the metric from the oracle's output using the regex in `metric_pattern`. It doesn't care about the rest of the output — the agent sees all of it, but the framework only extracts the metric value.

## Step 2: Write the starter solution

This is the baseline the agent starts from. It should be correct but unoptimized.

```python
# challenge/solution.py
def solve():
    """Naive solution. Returns a valid but unoptimized result."""
    return naive_result()
```

Run `verify.py` against it to get the baseline metric value — you'll need this for `reward.baseline` in the config.

## Step 3: Write the config

```toml
# problem.toml

[problem]
name = "my-problem"
system_prompt = """You are an optimization agent. Your goal is to maximize the score
by editing solution.py.

<describe the problem, constraints, and what the metric means>

Your tools:
  read_file(path)          — read files in the sandbox
  edit_file(path, content) — rewrite solution.py (full file, not a diff)
  run_eval()               — run the oracle, get your score
  run_command(command)      — shell commands (cp, ls, cat, diff, head, tail, wc, grep)

Strategy:
  1. Read solution.py and reference.py to understand the problem
  2. Form a hypothesis about how to improve
  3. Edit solution.py
  4. Run eval to check correctness and score
  5. If better: checkpoint with run_command('cp solution.py best.py')
     If worse: restore with run_command('cp best.py solution.py')
  6. Repeat

You have {max_attempts} attempts."""

[files]
challenge_dir = "challenge"           # relative to this TOML file
editable = ["solution.py"]            # files the agent can write
readable = ["solution.py", "verify.py", "reference.py"]  # files the agent can read

[oracle]
command = "python3 verify.py"         # run in the sandbox directory
metric_pattern = 'score=([\d.]+)'     # regex to extract metric from stdout
metric_name = "score"                 # name shown to agent and used in state
direction = "maximize"                # "maximize" or "minimize"
timeout_secs = 30                     # kill oracle after this many seconds

[reward]
type = "log"                          # "log" (log2 improvement) or "linear"
baseline = 1.0                        # metric value of the starter solution

[episode]
max_attempts = 20                     # optimization steps per episode
sandbox_pool_size = 4                 # concurrent sandboxes for parallel rollouts
```

### Config reference

**`[problem]`**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Identifier for the problem |
| `system_prompt` | yes | Prompt template. Can use `{max_attempts}`, `{metric_name}`, `{baseline_metric}` |

**`[files]`**

| Field | Default | Description |
|---|---|---|
| `challenge_dir` | `"challenge"` | Directory with challenge files, relative to TOML |
| `editable` | `[]` | Files the agent can write (full-file overwrite) |
| `readable` | `[]` | Files the agent can read (editable files are always readable too) |

**`[oracle]`**

| Field | Default | Description |
|---|---|---|
| `command` | required | Shell command to run in the sandbox |
| `metric_pattern` | required | Regex with one capture group for the metric value |
| `metric_name` | `"score"` | Name used in output and state tracking |
| `direction` | `"minimize"` | `"maximize"` or `"minimize"` |
| `timeout_secs` | `30` | Seconds before the oracle is killed |

**`[reward]`**

| Field | Default | Description |
|---|---|---|
| `type` | `"log"` | `"log"` = log2(improvement ratio), `"linear"` = normalized delta |
| `baseline` | `None` | Starting metric value. If None, read from dataset `info.baseline_metric` |
| `normalization` | `1.0` | For `"linear"` type: `reward = delta / normalization` |

**`[episode]`**

| Field | Default | Description |
|---|---|---|
| `max_attempts` | `20` | Max optimization steps (eval calls) per episode |
| `sandbox_pool_size` | `4` | Number of sandbox copies for parallel rollouts |
| `dataset_repeat` | `10000` | How many times to repeat the task in the dataset |

**`[plugins]`** (optional, for advanced use)

| Field | Description |
|---|---|
| `setup_hook` | `"module:function"` — called at episode start, before the agent acts |
| `reset_hook` | `"module:function"` — called when sandbox is reset between episodes |
| `extra_tools` | List of `"module:function"` — additional tools registered with the agent |

## Step 4: Test it

```python
# test_my_problem.py
import asyncio
from auto_rl import load_environment

env = load_environment("my_problem/problem.toml", sandbox_pool_size=1)

async def test():
    state = {"info": {"baseline_metric": 1.0}, "task": "test", "answer": ""}
    state = await env.setup_state(state)
    sid = state["sandbox_id"]

    # Verify tools work
    result = await env._tools["read_file"]("solution.py", sandbox_id=sid)
    print(result)

    result = await env._tools["run_eval"](sandbox_id=sid)
    print(result)

    await env.sandbox_pool.release(sid)

asyncio.run(test())
```

## Adding plugin hooks

For problems that need custom setup (e.g. installing a starting-point variant, running a baseline, GPU pinning):

```python
# my_hooks.py

async def setup_variant(sandbox_id, state, pool, config):
    """Install a starting-point variant based on task info."""
    import shutil
    variant = state.get("info", {}).get("variant_file", "baseline.py")
    sandbox = pool.get(sandbox_id)
    src = config.challenge_dir.parent / "variants" / variant
    shutil.copy2(src, sandbox.path / "solution.py")
```

```toml
# problem.toml
[plugins]
setup_hook = "my_hooks:setup_variant"
```

For extra tools:

```python
# my_hooks.py

async def run_analysis(script: str, sandbox_id: str) -> str:
    """Run an inline analysis script in the sandbox."""
    # ...
```

```toml
[plugins]
extra_tools = ["my_hooks:run_analysis"]
```

Extra tools must accept `sandbox_id: str` as a parameter — it's injected automatically and hidden from the agent's schema.

## Reward types

**`"log"`** — Best for problems with multiplicative improvement (e.g. cycle counts, runtime). A 2x improvement = reward 1.0, 4x = 2.0, 100x = 6.6. Good when early gains are easy and later gains are hard.

**`"linear"`** — Best for problems with additive improvement (e.g. accuracy, score). `reward = (best - baseline) / normalization`. Set `normalization` to what "good" looks like (e.g. if 0.02 improvement is strong, set `normalization = 0.02` so that gets reward 1.0).

Both types also include:
- **Monotonic improvement reward** (weight 0.25) — fraction of attempts that set a new best. Rewards systematic exploration over random thrashing.
- **Failure penalty** (weight 1.0) — `-0.05` per crash or invalid result.

## Example: circle packing

```
circle_packing/
  problem.toml
  challenge/
    solution.py    # returns list of (x, y, r) tuples
    verify.py      # checks non-overlap, in-bounds, prints score=sum_of_radii
    problem.py     # N=26, unit square constraints
```

```toml
[problem]
name = "circle-packing-26"
system_prompt = """You are an optimization agent. Pack 26 non-overlapping circles
into a unit square, maximizing the total sum of radii.

Edit solution.py to return a list of (x, y, radius) tuples.
Circles must not overlap each other or extend outside the [0,1] x [0,1] square.

Your tools:
  read_file(path)          — read solution.py or problem.py
  edit_file(path, content) — rewrite solution.py
  run_eval()               — check validity and get score
  run_command(command)      — shell commands (cp, ls, cat, diff, head, tail, wc, grep)

You have {max_attempts} attempts."""

[files]
challenge_dir = "challenge"
editable = ["solution.py"]
readable = ["solution.py", "problem.py", "verify.py"]

[oracle]
command = "python3 verify.py"
metric_pattern = 'score=([\d.]+)'
metric_name = "score"
direction = "maximize"
timeout_secs = 10

[reward]
type = "log"
baseline = 0.5

[episode]
max_attempts = 20
```

## Training

```bash
pip install -e .  # install auto-rl

# Use with prime-rl (see kernel/rl.toml for an example config)
prime-rl train --config rl.toml
```

The `rl.toml` references the environment via its entry point:

```toml
[orchestrator.env]
id = "my-problem"

[orchestrator.env.env_args]
max_attempts = 20
sandbox_pool_size = 8
```
