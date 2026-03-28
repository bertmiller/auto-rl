# RL Environment Spec: `kernel-opt-env`
### Training models to optimize kernels on a custom VLIW SIMD architecture
### Stack: `verifiers` + `prime-rl`

---

## 1. What the task is

The kernel optimization challenge is Anthropic's original performance
engineering take-home. The agent must optimize a kernel running on a simulated
custom VLIW SIMD machine. The concrete steps per iteration are:

1. Read `perf_takehome.py` (the `KernelBuilder.build_kernel` method and helpers)
2. Read `problem.py` (the machine simulator, ISA reference, and problem definition)
3. Form a hypothesis about what optimization to apply
4. Edit `perf_takehome.py` to implement the optimization
5. Run `python tests/submission_tests.py` to verify correctness and measure cycles
6. If improved and correct → keep the change. If not → revert.
7. Repeat.

The oracle is a deterministic Python simulator of a custom VLIW SIMD machine.
The action is a code edit to `perf_takehome.py`. The state is the full content
of `perf_takehome.py` plus the history of prior optimization attempts. The
metric is **clock cycles** — lower is better.

### The problem

The kernel performs a parallel tree traversal: a batch of 256 inputs traverses
a binary tree of height 10 for 16 rounds. At each node, the input value is
XORed with the node value, hashed through a 6-stage hash function, and the
hash result determines left/right branching. If the traversal reaches a leaf,
it wraps back to the root.

### The machine

The simulated machine is a VLIW (Very Long Instruction Word) SIMD architecture:

- **Engines:** `alu` (12 slots), `valu` (6 slots), `load` (2 slots),
  `store` (2 slots), `flow` (1 slot) — all execute in parallel per cycle
- **SIMD:** Vector operations process `VLEN=8` elements per slot
- **Memory model:** Effects don't take effect until end of cycle; all inputs
  are read before any writes land
- **Scratch space:** 1536 words serving as registers, constants, and cache
- **Single core** (`N_CORES=1`)

The baseline starter code is a naive scalar implementation that processes one
batch element at a time with no instruction-level parallelism, no SIMD, and
no loop constructs — the entire computation is fully unrolled. It runs in
**147,734 cycles**.

### Key optimization axes

- **VLIW packing:** Fill multiple slots per instruction bundle instead of one
- **SIMD vectorization:** Use `valu`, `vload`, `vstore` to process 8 elements per slot
- **Loop construction:** Use `jump`/`cond_jump` instead of unrolling
- **Instruction scheduling:** Reorder to maximize parallel slot utilization
- **Memory access patterns:** Batch loads/stores, use vector memory ops
- **Algorithmic:** Precompute, reuse intermediate values, reduce redundant work

---

## 2. MDP (Markov Decision Process)

```
S (State space) = (perf_takehome_py_content: str,
                   problem_py_content: str,          # read-only reference
                   optimization_history: list[OptimizationResult],
                   current_cycles: int)

A (Action space) = sequence of tool calls over one episode:
                     read_file(path)
                     edit_file(path, content)
                     run_tests()
                     run_analysis(script)

T (Transition function) = deterministic — the simulator is fully deterministic
                           given the same random seed, so identical code always
                           produces identical cycle counts.

R (Reward function) = log2(baseline_cycles / best_cycles)           [primary: speedup]
                      + 0.25 × monotonic_improvement_fraction       [process]
                      − 0.05 × num_failures                         [penalty]

γ (Discount factor) = 0.99

H (Horizon) = 20 optimization attempts per episode
              (max_turns = 80, covering ~4 tool calls per optimization cycle)
```

**Episode initialization.** Every episode starts from the identical baseline:
the unoptimized `KernelBuilder.build_kernel` producing 147,734 cycles. A fresh
copy of the working directory is created per rollout for isolation.

Variation across rollouts comes from sampling temperature alone — the same
starting point, different sequences of optimization ideas. This is intentional:
the model must learn general kernel optimization strategy, not memorize a
fixed optimization sequence.

**Why log2 speedup?** Cycle reduction is multiplicative — going from 147k to
73k (2×) is roughly as hard as going from 73k to 36k (another 2×). Log2 makes
the reward scale linear in difficulty. A 2× speedup = reward 1.0,
4× = 2.0, 100× = 6.6.

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
        path: File path relative to the challenge root.
              Use 'perf_takehome.py' to read the kernel code,
              'problem.py' to read the simulator and ISA reference,
              'test_output.log' to read the last test run output.

    Returns:
        Full file content as a string.
    """
    ALLOWED_PATHS = [
        "perf_takehome.py", "problem.py",
        "test_output.log", "tests/submission_tests.py",
    ]
    if path not in ALLOWED_PATHS:
        return f"Error: cannot read '{path}'. Allowed: {ALLOWED_PATHS}"
    return await sandbox_read_file(sandbox_id, path)


async def edit_file(
    path: str,
    content: str,
    sandbox_id: str,  # injected
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


async def run_tests(
    sandbox_id: str,  # injected
) -> str:
    """
    Run the submission tests and return correctness + cycle count.
    Uses the frozen simulator in tests/frozen_problem.py to prevent
    cheating by modifying the simulator.

    Returns:
        Success: "CORRECT | cycles=18532 | speedup=7.97x | tests_passed=3/8"
        Failure: "INCORRECT | error: Incorrect output values"
        Crash:   "CRASH | last 50 lines: ..."
    """
    result = await sandbox_run_tests(sandbox_id)
    return format_test_result(result)


async def run_analysis(
    script: str,
    sandbox_id: str,  # injected
) -> str:
    """
    Run an inline Python analysis script to inspect the kernel's behavior.
    The script executes in the sandbox with access to problem.py and
    perf_takehome.py imports. Use this to analyze bottlenecks, inspect
    instruction schedules, count slot utilization, profile cycle
    distribution, etc.

    The script runs with a 30-second timeout. It cannot modify
    perf_takehome.py or any other file — only read and compute.

    Args:
        script: A complete Python script as a string. It can import from
                problem.py and perf_takehome.py. Print results to stdout.

    Returns:
        stdout output (truncated to 10,000 chars), or an error message
        if the script crashes or times out.

    Example script:
        '''
        from perf_takehome import KernelBuilder
        from problem import Machine, Tree, Input, build_mem_image, N_CORES, SLOT_LIMITS
        import random

        random.seed(123)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)

        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

        # Count how many slots are filled per instruction bundle
        slot_counts = {name: 0 for name in SLOT_LIMITS if name != "debug"}
        total_instrs = 0
        for instr in kb.instrs:
            non_debug = {k: v for k, v in instr.items() if k != "debug"}
            if non_debug:
                total_instrs += 1
                for name, slots in non_debug.items():
                    slot_counts[name] += len(slots)

        print(f"Total instruction bundles: {total_instrs}")
        print(f"Slot utilization:")
        for name, count in slot_counts.items():
            limit = SLOT_LIMITS[name]
            print(f"  {name}: {count}/{total_instrs * limit} "
                  f"({100 * count / (total_instrs * limit):.1f}%)")
        '''
    """
    result = await sandbox_run_analysis(sandbox_id, script, timeout_secs=30)
    return format_analysis_result(result)
```

**Why full-file `edit_file`?** Same rationale as autoresearch: diffs require
precise line offsets which models frequently get wrong. `perf_takehome.py` is
~276 lines — small enough for full-file rewrites. The model reads the current
file, modifies it in context, and writes the complete result.

**Why restrict file access?** The README explicitly warns that LLMs cheat by
modifying `tests/` or `problem.py` (e.g. changing `N_CORES=1` to get fake
multicore speedups). The environment enforces read-only access on everything
except `perf_takehome.py`, and uses `tests/frozen_problem.py` for evaluation.

---

## 4. Environment class

```python
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset
import uuid
import subprocess
import re


class KernelOptEnv(vf.StatefulToolEnv):
    """
    RL environment for training models to optimize kernels on a custom
    VLIW SIMD architecture.

    One episode = one optimization session. The agent iterates on
    perf_takehome.py for up to max_attempts optimization cycles,
    starting from the same naive baseline every time.

    Per-rollout state:
        sandbox_id: str               — isolated working directory
        baseline_cycles: int           — 147734 (naive scalar implementation)
        best_cycles: int               — best cycle count this episode
        current_cycles: int            — most recent cycle count
        optimization_history: list     — [{step, cycles, speedup, correct, description}]
        num_failures: int              — incorrect results or crashes
        episode_done: bool
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_attempts: int = 20,
        sandbox_pool_size: int = 64,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            tools=[read_file, edit_file, run_tests, run_analysis],
            max_turns=max_attempts * 4,
            args_to_skip=["sandbox_id"],
            **kwargs,
        )
        self.sandbox_pool = KernelOptSandboxPool(size=sandbox_pool_size)

    async def setup_state(self, state: State) -> State:
        """
        Allocate a fresh sandbox with a clean copy of the challenge files.
        No baseline run needed — the baseline cycle count is a known constant.
        """
        sandbox_id = await self.sandbox_pool.acquire()

        # Reset to clean baseline
        await sandbox_reset(sandbox_id)

        state.update({
            "sandbox_id": sandbox_id,
            "baseline_cycles": 147734,
            "best_cycles": 147734,
            "current_cycles": 147734,
            "optimization_history": [{
                "step": 0,
                "cycles": 147734,
                "speedup": 1.0,
                "correct": True,
                "description": "baseline (naive scalar, no ILP, no SIMD)",
            }],
            "num_failures": 0,
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
import math


async def speedup_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Primary reward: log2 speedup over baseline.

    log2(147734 / best_cycles)

    Reference points:
      - 2× speedup (73867 cycles) → reward 1.0
      - 8× speedup (18467 cycles) → reward 3.0  (updated starting point level)
      - 68× speedup (2164 cycles)  → reward 6.1  (Opus 4 many-hours level)
      - 100× speedup (1477 cycles) → reward 6.6  (Opus 4.5 11.5hr level)

    Failure penalty subtracted.
    """
    baseline = state.get("baseline_cycles", 147734)
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

    Rewards systematic optimization. Penalizes random thrashing —
    an agent that tries random edits and occasionally stumbles into
    improvements will score poorly here.
    """
    history = [
        e for e in state.get("optimization_history", [])
        if e["correct"] and e["step"] > 0
    ]
    if not history:
        return 0.0
    improvements = sum(1 for e in history if e["cycles"] < state["baseline_cycles"]
                       and all(prev["cycles"] >= e["cycles"]
                               for prev in history if prev["step"] < e["step"]))
    return improvements / len(history)


rubric = vf.Rubric(
    funcs=[
        speedup_reward,
        monotonic_improvement_reward,
    ],
    weights=[1.0, 0.25],
)
```

---

## 6. Dataset

The dataset is a single fixed task, repeated. All variation comes from
sampling across rollouts.

```python
TASK = {
    "task_id": "kernel-opt-baseline",
    "prompt": SYSTEM_PROMPT,
    "answer": 1487,    # reference: Opus 4.5 after 11.5 hours
    "info": {
        "baseline_cycles": 147734,
        "target_thresholds": [18532, 2164, 1790, 1579, 1548, 1487, 1363],
    },
}

dataset = Dataset.from_list([TASK] * DATASET_REPEAT_N)
# DATASET_REPEAT_N = 10_000; never exhausted across a training run.
```

The `answer` field (1487) is a soft normalization anchor — a model reaching
this level is performing at Opus 4.5's 11.5-hour level.

---

## 7. System prompt

```python
SYSTEM_PROMPT = """\
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

The optimization loop:
  1. Read perf_takehome.py and problem.py to understand the current state.
  2. Use run_analysis() to profile the kernel — inspect slot utilization,
     cycle distribution, instruction counts, or any other metric.
  3. Identify the bottleneck and form an optimization hypothesis.
  4. Edit perf_takehome.py to implement it.
  5. Run tests to verify correctness and measure cycles.
  6. If correct and faster: keep. If incorrect or slower: revert to the
     last known-good version and try something else.
  7. Repeat.

Do NOT modify problem.py or anything in tests/. The submission tests use
a frozen copy of the simulator.

Optimization strategies to consider:
  - VLIW instruction packing (fill multiple slots per bundle)
  - SIMD vectorization (process 8 batch elements per vector op)
  - Loop construction with jump/cond_jump instead of full unrolling
  - Instruction scheduling to maximize parallelism
  - Precomputation and constant folding

You have 20 optimization attempts. Be systematic — understand what the
bottleneck is before changing code.
"""
```

---

## 8. `prime-rl` training config (`rl.toml`)

```toml
[trainer]
total_steps = 5000
batch_size = 64
micro_batch_size = 4
gradient_accumulation_steps = 16
learning_rate = 5e-7
lr_scheduler = "cosine"
warmup_steps = 150
max_grad_norm = 0.5
algorithm = "grpo"
grpo_num_generations = 8       # 8 rollouts per seed; cheap oracle allows more
grpo_epsilon = 0.2
grpo_epsilon_high = 0.25
use_advantage_normalization = true
use_difficulty_buffer = false  # single fixed task

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
max_model_len = 32768          # perf_takehome.py + problem.py + history fits easily
enable_auto_tool_choice = true
tool_call_parser = "hermes"
gpu_memory_utilization = 0.85

[inference.sampling]
temperature = 0.8
top_p = 0.95
max_tokens = 8192              # full perf_takehome.py rewrite + reasoning

[orchestrator]
num_workers = 64               # cheap oracle → can run many parallel rollouts
max_pending_rollouts = 256
rollout_timeout_secs = 600     # 10min: 20 attempts × ~1s each + overhead

[[orchestrator.env]]
id = "kernel-opt/vliw-simd"
args = { max_attempts = 20, sandbox_pool_size = 64 }

[logging]
wandb_project = "kernel-opt-rl"
log_every_n_steps = 5
eval_every_n_steps = 100
eval_num_examples = 32         # cheap to eval — just Python simulation
```

---

## 9. Sandbox requirements

Each rollout needs only a CPU Python process — no GPU required. Sandboxes
must be pre-provisioned with:

- A clean copy of the challenge directory (`problem.py`, `perf_takehome.py`,
  `tests/`, `watch_trace.py`, `watch_trace.html`)
- Python 3.10+ with no additional dependencies (stdlib only)
- `tests/frozen_problem.py` is a frozen copy of `problem.py` used by the
  submission tests — prevents cheating via simulator modification

**Isolation:** Each sandbox is a separate directory (or container). The agent
can only modify `perf_takehome.py`. The submission tests import `KernelBuilder`
from `perf_takehome.py` but use `frozen_problem.py` for the `Machine` simulator.

**Throughput:** This is a dramatically cheaper oracle than autoresearch.
A single test run takes ~1–5 seconds (pure Python simulation), not 5 minutes
of GPU training. At 64 workers, 20 attempts per episode, ~3s per test run:
each episode completes in ~60s. That is ~3800 completed rollouts/hour — orders
of magnitude faster than autoresearch. At `grpo_num_generations=8`, that is
~475 training steps/hour. A full 5000-step training run completes in ~10 hours
on a single training node.

**No baseline run needed at episode start.** The baseline cycle count (147,734)
is a known constant derived from the starter code. Unlike autoresearch, there
is no stochastic oracle to calibrate against — the simulator is deterministic.

---

## 10. Open questions

**Determinism is a double-edged sword.** The simulator is fully deterministic:
identical code always produces identical cycle counts. This means the reward
signal has zero noise, which is excellent for training stability. But it also
means the model can memorize exact optimization sequences without understanding
why they work. Mitigations: (1) the space of valid optimizations is large
enough that memorization is unlikely to capture the full Pareto frontier,
(2) the log2 reward scale means diminishing returns push the model toward
deeper understanding for the last few bits of speedup.

**Reward shaping at different skill levels.** The gap between baseline
(147,734 cycles) and state-of-the-art (~1,363 cycles) is enormous — a 108×
speedup. Early in training, any improvement (e.g. basic loop construction for
8×) yields large reward. Late in training, squeezing out the last 2× requires
sophisticated instruction scheduling. The log2 reward function handles this
naturally, but consider whether a curriculum (e.g. start with smaller problems
before the full height=10, rounds=16, batch=256 challenge) would accelerate
early learning.

**SFT cold start.** The initial policy will not know the ISA or how to write
valid instruction bundles. Consider bootstrapping with SFT on:
- The problem description and ISA documentation extracted from `problem.py`
- Example optimizations with explanations (e.g. "vectorize the hash stage
  using valu ops, packing 8 batch elements per vector")
- Traces of successful optimization sessions from Claude models

~100–200 SFT steps teaching the model the basic ISA and how to construct valid
`KernelBuilder` code should be sufficient to get non-trivial reward variance
for GRPO to work with.

**Cheating prevention.** The environment restricts edits to `perf_takehome.py`
and uses `frozen_problem.py` for evaluation. But the model could still attempt
subtle cheats within `perf_takehome.py` itself — e.g. detecting the test
harness and returning hardcoded results. The submission tests use random
(unseeded) inputs across 8 correctness trials, making this difficult but not
impossible. Consider adding a post-episode verification step that runs the
kernel on adversarially chosen inputs.

**Scaling to harder variants.** Once the model saturates on this problem,
natural extensions include: larger batch sizes, deeper trees, multi-core
variants (re-enabling `N_CORES > 1`), or entirely different kernels on the
same ISA (matrix multiply, convolution, sorting). This would test whether
the model has learned general VLIW SIMD optimization rather than memorizing
tree-traversal-specific tricks.
