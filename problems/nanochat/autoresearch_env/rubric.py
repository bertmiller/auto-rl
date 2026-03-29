"""
Reward functions for the autoresearch RL environment.

Primary reward: total val_bpb improvement over baseline.
Process reward: fraction of experiments that set a new best.
"""

from __future__ import annotations

import verifiers as vf


async def val_bpb_improvement_reward(
    completion: list[dict],
    state: dict,
    **kwargs,
) -> float:
    """
    Primary reward: total val_bpb improvement over baseline this episode.
    Normalized so that 0.02 improvement (a strong session) -> reward ~1.0.
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

    Rewards efficient hill-climbing. Penalizes random thrashing --
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
