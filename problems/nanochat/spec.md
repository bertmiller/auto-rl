# RL Environment Spec: `autoresearch-env`
### Training models to do better at karpathy/autoresearch
### Stack: `verifiers` + `prime-rl`

---

## 1. What the task is

The autoresearch loop has five concrete steps per experiment:

1. Read `train.py` (630 lines: GPT model, Muon+AdamW optimizer, training loop)
2. Form a hypothesis about what to change
3. Edit `train.py` directly (free-form code modification)
4. Run `uv run train.py > run.log 2>&1`, wait 5 minutes
5. Extract result: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If improved → `git commit`. If not → `git reset --hard HEAD`.
7. Repeat.

The oracle is a real GPU training run. The action is a code edit. The state is
the full content of `train.py` plus the history of prior experiments in this
episode. The metric is `val_bpb` (validation bits per byte) — lower is better,
vocab-size-independent, so architectural changes are fairly compared.

---

## 2. MDP

```
S  = (train_py_content: str,
      experiment_history: list[ExperimentResult],
      program_md: str,
      baseline_val_bpb: float)

A  = sequence of tool calls over one episode:
       read_file(path)
       edit_file(path, content)
       run_experiment()
       git_commit_or_revert(action, message)

T  = deterministic given action; stochastic only through oracle noise
     (~±0.0002 val_bpb variance on identical runs)

R  = (baseline_val_bpb − best_val_bpb_this_episode)   [primary]
     + 0.25 × monotonic_improvement_fraction           [process]
     − 0.001 × num_crashes                             [penalty]

γ  = 0.99

H  = 15 experiments per episode
     (max_turns = 60, covering ~4 tool calls per experiment cycle)
```

**Episode initialization.** Every episode starts from the identical baseline:
HEAD of `karpathy/autoresearch`, unmodified `train.py`, unmodified `program.md`.
A fresh git branch (`autoresearch/rl-episode-{uuid}`) is created per rollout for
isolation. The run-0 baseline score is established by running `train.py` once
at the start of `setup_state` before the model receives any context.

Variation across rollouts comes from sampling temperature alone — the same
starting point, different sequences of hypotheses. This is intentional: the
model must generalize its research strategy, not memorize a task distribution.

---

## 3. Tools

All tools are async Python functions. `sandbox_id` is injected by
`StatefulToolEnv` and hidden from the model's schema via `args_to_skip`.

```python
async def read_file(
    path: str,
    sandbox_id: str,  # injected
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
    sandbox_id: str,  # injected
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


async def run_experiment(
    sandbox_id: str,  # injected
) -> str:
    """
    Run the training experiment and return the result.
    Training runs for a fixed 5-minute wall-clock budget.

    Returns:
        Success: "val_bpb=0.9847 | peak_vram_mb=44210 | training_seconds=300.1"
        Crash:   "CRASH | last 50 lines of traceback: ..."
    """
    result = await sandbox_run_train(sandbox_id, train_secs=300)
    return format_run_result(result)


async def git_commit_or_revert(
    action: str,
    message: str = "",
    sandbox_id: str,  # injected
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
```

**Why full-file `edit_file` instead of a diff tool?** Diffs require the model
to emit precise line offsets which it frequently gets wrong. Full-file rewrites
are unambiguous — the model reads the current file, modifies it in context, and
writes the complete result. At 630 lines, `train.py` is small enough that this
is practical.

---

## 4. Environment class

```python
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset
import uuid


class AutoresearchEnv(vf.StatefulToolEnv):
    """
    RL environment for training models to do better at karpathy/autoresearch.

    One episode = one research session. The agent iterates on train.py
    for up to max_experiments cycles, starting from the same baseline
    every time.

    Per-rollout state:
        sandbox_id: str              — isolated container with dedicated GPU
        baseline_val_bpb: float      — score from run-0 before any edits
        best_val_bpb: float          — best score seen this episode
        experiment_history: list     — [{step, val_bpb, delta, crashed, message}]
        num_crashes: int
        episode_done: bool
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_experiments: int = 15,
        sandbox_pool_size: int = 16,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            tools=[read_file, edit_file, run_experiment, git_commit_or_revert],
            max_turns=max_experiments * 4,  # ~4 tool calls per experiment cycle
            args_to_skip=["sandbox_id"],
            **kwargs,
        )
        self.sandbox_pool = AutoresearchSandboxPool(size=sandbox_pool_size)

    async def setup_state(self, state: State) -> State:
        """
        Allocate a fresh sandbox, create an isolated branch,
        and establish the baseline val_bpb by running train.py once
        before the model receives any context.
        """
        sandbox_id = await self.sandbox_pool.acquire()

        # Isolate this rollout on its own branch
        branch = f"autoresearch/rl-episode-{uuid.uuid4().hex[:8]}"
        await sandbox_git_checkout_branch(sandbox_id, branch)

        # Establish baseline (run-0)
        baseline_result = await sandbox_run_train(sandbox_id, train_secs=300)
        baseline_val_bpb = baseline_result["val_bpb"]

        state.update({
            "sandbox_id": sandbox_id,
            "baseline_val_bpb": baseline_val_bpb,
            "best_val_bpb": baseline_val_bpb,
            "experiment_history": [{
                "step": 0,
                "val_bpb": baseline_val_bpb,
                "delta": 0.0,
                "crashed": False,
                "message": "baseline",
            }],
            "num_crashes": 0,
            "episode_done": False,
        })
        return await super().setup_state(state)

    def update_tool_args(self, tool_name: str, args: dict, state: State) -> dict:
        args["sandbox_id"] = state["sandbox_id"]
        return args

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    @vf.cleanup
    async def release_sandbox(self, state: State):
        if sid := state.get("sandbox_id"):
            await self.sandbox_pool.release(sid)
```

---

## 5. Rubric

```python
async def val_bpb_improvement_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Primary reward: total val_bpb improvement over baseline this episode.
    Normalized so that 0.02 improvement (a strong session) → reward ~1.0.
    Crash penalty subtracted.

    Reference: Karpathy's overnight runs achieved ~0.02 improvement
    over ~90 experiments. A 15-experiment session achieving half that
    (~0.01) is a good result.
    """
    baseline = state.get("baseline_val_bpb", 1.0)
    best = state.get("best_val_bpb", baseline)
    improvement = baseline - best
    crash_penalty = state.get("num_crashes", 0) * 0.001
    normalized = improvement / 0.02
    return float(max(0.0, min(normalized - crash_penalty, 2.0)))


async def monotonic_improvement_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Process reward: fraction of non-baseline, non-crash experiments
    that set a new best val_bpb.

    Rewards efficient hill-climbing. Penalizes random thrashing —
    an agent that tries random edits and occasionally gets lucky on
    the primary reward will score poorly here.
    """
    history = [
        e for e in state.get("experiment_history", [])
        if not e["crashed"] and e["step"] > 0
    ]
    if not history:
        return 0.0
    improvements = sum(1 for e in history if e["delta"] > 0)
    return improvements / len(history)


rubric = vf.Rubric(
    funcs=[
        val_bpb_improvement_reward,
        monotonic_improvement_reward,
    ],
    weights=[1.0, 0.25],
)
```

---

## 6. Dataset

The dataset is a single fixed task, repeated. All variation comes from
sampling across rollouts — there is no task distribution to draw from.

```python
TASK = {
    "task_id": "autoresearch-baseline",
    "prompt": SYSTEM_PROMPT,
    "answer": 0.975,   # reference: Karpathy's best from ~90-experiment sessions
    "info": {
        "baseline_commit": "HEAD",   # karpathy/autoresearch main branch
    }
}

dataset = Dataset.from_list([TASK] * DATASET_REPEAT_N)
# DATASET_REPEAT_N should be large enough that the dataloader never exhausts
# across a training run; 10_000 is sufficient.
```

The `answer` field (0.975) is used as a soft normalization anchor in the
rubric — a model that reaches this target in 15 experiments is performing
at human-expert level for a single overnight session.

---

## 7. System prompt

```python
SYSTEM_PROMPT = """\
You are an ML research agent. Your task is to improve the val_bpb
(validation bits per byte) of a small GPT training script by running
experiments autonomously.

Lower val_bpb is better. The training script runs for exactly 5 minutes
per experiment regardless of what you change.

Your tools:
  read_file(path)                      — read train.py, run.log, or results.tsv
  edit_file(path, content)             — rewrite train.py with your changes
  run_experiment()                     — run training and get val_bpb
  git_commit_or_revert(action, msg)    — keep or discard the change

The research loop:
  1. Read train.py to understand the current state.
  2. Form a hypothesis. Change one thing at a time.
  3. Edit train.py to implement it.
  4. Run the experiment.
  5. If val_bpb improved: commit. If not: revert.
  6. Repeat.

If a run crashes, read run.log to diagnose. Attempt one fix.
If the fix also crashes, revert and try a different idea.

You have 15 experiments. Use them to make meaningful, targeted progress.
"""
```

---

## 8. `prime-rl` training config (`rl.toml`)

```toml
[trainer]
total_steps = 3000
batch_size = 32
micro_batch_size = 2
gradient_accumulation_steps = 16
learning_rate = 5e-7
lr_scheduler = "cosine"
warmup_steps = 100
max_grad_norm = 0.5
algorithm = "grpo"
grpo_num_generations = 4       # 4 rollouts per seed; oracle GPUs are the bottleneck
grpo_epsilon = 0.2
grpo_epsilon_high = 0.25
use_advantage_normalization = true
use_difficulty_buffer = false  # single fixed task; difficulty buffer not applicable

[trainer.model]
name = "Qwen/Qwen3-14B-Instruct"
enable_lora = true
lora_rank = 128
lora_alpha = 256
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

[inference]
model = "Qwen/Qwen3-14B-Instruct"
enable_lora = true
tensor_parallel_size = 2
max_model_len = 32768          # full train.py + history fits comfortably
enable_auto_tool_choice = true
tool_call_parser = "hermes"
gpu_memory_utilization = 0.85

[inference.sampling]
temperature = 0.8              # high enough to get diverse edits across 4 rollouts
top_p = 0.95
max_tokens = 4096              # enough for a full train.py rewrite per turn

[orchestrator]
num_workers = 16               # 16 parallel rollouts, each on its own GPU
max_pending_rollouts = 64
rollout_timeout_secs = 5400    # 90min: 15 × 300s + sandbox overhead

[[orchestrator.env]]
id = "autoresearch/nanochat-opt"
args = { max_experiments = 15, sandbox_pool_size = 16 }

[logging]
wandb_project = "autoresearch-rl"
log_every_n_steps = 5
eval_every_n_steps = 100
eval_num_examples = 8          # 8 full 300s-per-experiment episodes = ~6 GPU-hours
```

---

## 9. Sandbox requirements

Each rollout needs a dedicated GPU for the duration of the episode. Sandboxes
must be pre-provisioned with:

- `karpathy/autoresearch` cloned at HEAD
- `uv sync` completed
- `uv run prepare.py` completed (TinyStories data + tokenizer)
- PyTorch + CUDA ready, no cold-start compilation cost

At episode start, `setup_state` runs one baseline training job (~5min) before
returning control to the model. This is the most expensive part of episode
initialization and cannot be parallelized with rollout generation.

**Throughput:** 16 oracle GPUs × 1 rollout each, each episode taking ~90min
(1 baseline run + 15 experiment runs at 5min each) = ~10 completed rollouts/hour.
At `grpo_num_generations=4`, that is ~2.5 training steps/hour. Plan for
multi-day training runs.

---

## 10. Open questions

**Oracle noise vs. signal.** Val_bpb variance on identical runs is ~±0.0002.
The smallest meaningful single-experiment improvement is ~0.0005 — a 2.5×
signal-to-noise ratio. This is real but tight. If advantage estimates are
noisy early in training, consider running each experiment twice and reporting
the mean. This halves throughput but stabilizes the reward signal.

**SFT cold start.** GRPO from scratch on free-form code generation produces
near-zero reward variance early in training: the initial policy generates
mostly invalid edits, crashes dominate, and the advantage signal collapses.
Karpathy's GitHub Discussions contain full logs of autonomous sessions —
extract (train.py_before, run_result, train.py_after) triples from committed
experiments and use them as multi-turn SFT demonstration data before starting RL.
~200 SFT steps is sufficient to teach the model the basic loop mechanics.

**Fixed baseline overfitting.** With no task variation, the model may memorize
a fixed sequence of high-value edits (e.g. "always halve batch size on step 1")
rather than learning general research strategy. This is acceptable for a first
training run. The natural extension once the loop is working: use checkpoints
from different points in Karpathy's experiment history as varied baselines —
the model must then adapt to a codebase that is already partially optimized
rather than always starting from the same blank slate.
