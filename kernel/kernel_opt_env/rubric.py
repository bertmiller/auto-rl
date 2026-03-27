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
