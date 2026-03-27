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

The kernel performs a batched tree traversal with hashing. Your starting
implementation runs in {baseline_cycles:,} cycles. Your goal is to reduce
this as much as possible.

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

# Starting points: (variant_filename, baseline_cycles)
# Each is a perf_takehome.py at a different optimization level.
# The original baseline is the naive scalar implementation.
# Variants come from prior optimization sessions at various stages.
STARTING_POINTS = [
    ("baseline_147734.py", 147734),
    ("variant_15973.py", 15973),
    ("variant_5780.py", 5780),
    ("variant_5356.py", 5356),
    ("variant_4237.py", 4237),
    ("variant_3157.py", 3157),
    ("variant_2786.py", 2786),
    ("variant_2432.py", 2432),
]


def _build_dataset(system_prompt: str) -> Dataset:
    tasks = []
    for variant_file, baseline_cycles in STARTING_POINTS:
        task = {
            "task_id": f"kernel-opt-{baseline_cycles}",
            "prompt": system_prompt,
            "answer": 1487,  # Opus 4.5 after 11.5 hours
            "info": {
                "baseline_cycles": baseline_cycles,
                "variant_file": variant_file,
            },
        }
        tasks.append(task)
    # Repeat the task list to fill the dataset
    full = tasks * (DATASET_REPEAT_N // len(tasks) + 1)
    return Dataset.from_list(full[:DATASET_REPEAT_N])


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
    # Use a placeholder prompt for dataset construction; the env
    # formats the real prompt per-episode in setup_state using
    # the task's baseline_cycles.
    placeholder_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        max_attempts=max_attempts, baseline_cycles=147734
    )
    dataset = _build_dataset(placeholder_prompt)

    return KernelOptEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        max_attempts=max_attempts,
        sandbox_pool_size=sandbox_pool_size,
        **kwargs,
    )
