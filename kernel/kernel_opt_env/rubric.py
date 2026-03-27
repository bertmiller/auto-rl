"""
Reward functions for the kernel optimization RL environment.

Primary reward: log2 speedup over per-task baseline, scaled by difficulty.
Process reward: fraction of attempts that set a new best.
"""

from __future__ import annotations

import math

import verifiers as vf


BASELINE_CYCLES = 147734


def _difficulty_multiplier(baseline_cycles: int) -> float:
    """
    Scale reward by how hard the starting point is.

    Harder variants (lower starting cycles) get a higher multiplier
    so improvements on them produce comparable absolute reward to
    large improvements on easy variants.

    multiplier = log2(147734 / baseline_cycles) + 1

    Examples:
      baseline 147734 -> 1.0
      variant  15973  -> ~4.2
      variant   2432  -> ~6.9
    """
    if baseline_cycles >= BASELINE_CYCLES:
        return 1.0
    return math.log2(BASELINE_CYCLES / baseline_cycles) + 1


async def speedup_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Primary reward: log2 speedup over per-task baseline, scaled by difficulty.

    reward = log2(baseline / best) * difficulty_multiplier - failure_penalty

    The difficulty multiplier ensures that small improvements on hard
    variants produce similar absolute reward to large improvements on
    easy variants.

    Failure penalty: 0.05 per failure subtracted.
    """
    baseline = state.get("baseline_cycles", BASELINE_CYCLES)
    best = state.get("best_cycles", baseline)
    failure_penalty = state.get("num_failures", 0) * 0.05
    multiplier = _difficulty_multiplier(baseline)

    if best >= baseline:
        return max(0.0, -failure_penalty)

    raw = math.log2(baseline / best) * multiplier
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


async def notes_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Small reward for maintaining notes.md.

    Checks that notes.md exists and has been updated with substantive
    content (not just empty or a single line). Rewards scale with
    the number of optimization attempts documented.
    """
    from .sandbox import sandbox_read_file

    sid = state.get("sandbox_id")
    if not sid:
        return 0.0

    content = await sandbox_read_file(sid, "notes.md")
    if content.startswith("Error:"):
        return 0.0

    lines = [l for l in content.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return 0.0

    # Reward scales with content up to a cap
    return min(len(lines) / 20, 1.0)


rubric = vf.Rubric(
    funcs=[
        speedup_reward,
        monotonic_improvement_reward,
        notes_reward,
    ],
    weights=[1.0, 0.25, 0.1],
)
