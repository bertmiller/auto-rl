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
