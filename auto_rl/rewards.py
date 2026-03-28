from __future__ import annotations

import math

import verifiers as vf


async def log_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    baseline = state["baseline_metric"]
    best = state["best_metric"]
    direction = state["metric_direction"]

    if direction == "minimize":
        if best >= baseline:
            return 0.0
        ratio = baseline / best
    else:
        if best <= baseline:
            return 0.0
        ratio = best / baseline

    return float(min(math.log2(ratio), 10.0))


async def linear_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    baseline = state["baseline_metric"]
    best = state["best_metric"]
    direction = state["metric_direction"]
    normalization = state.get("reward_normalization", 1.0)

    if direction == "minimize":
        delta = baseline - best
    else:
        delta = best - baseline

    return float(max(0.0, min(delta / normalization, 10.0)))


async def monotonic_improvement_reward(completion: list, state: dict, **kwargs) -> float:
    history = state.get("optimization_history", [])
    direction = state.get("metric_direction", "minimize")
    non_baseline = [e for e in history if e.get("correct") and e.get("step", 0) > 0 and e.get("metric") is not None]
    if not non_baseline:
        return 0.0

    improvements = 0
    best_so_far = state["baseline_metric"]
    for e in non_baseline:
        m = e["metric"]
        if (direction == "minimize" and m < best_so_far) or \
           (direction == "maximize" and m > best_so_far):
            improvements += 1
            best_so_far = m

    return improvements / len(non_baseline)


async def failure_penalty(completion: list, state: dict, **kwargs) -> float:
    return -0.05 * state.get("num_failures", 0)


def build_rubric(
    reward_type: str = "log",
    weights: dict[str, float] | None = None,
) -> vf.Rubric:
    weights = weights or {}
    improvement_fn = log_improvement_reward if reward_type == "log" else linear_improvement_reward
    funcs = [improvement_fn, monotonic_improvement_reward, failure_penalty]
    w = [
        weights.get("improvement", 1.0),
        weights.get("monotonic", 0.25),
        weights.get("failure", 1.0),
    ]
    return vf.Rubric(funcs=funcs, weights=w)
