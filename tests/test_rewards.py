import asyncio
from auto_rl.rewards import (
    log_improvement_reward,
    linear_improvement_reward,
    monotonic_improvement_reward,
    failure_penalty,
    build_rubric,
)

def _state(baseline, best, direction="minimize", failures=0, history=None):
    return {
        "baseline_metric": baseline,
        "best_metric": best,
        "metric_direction": direction,
        "reward_normalization": 1.0,
        "num_failures": failures,
        "optimization_history": history or [],
    }

def test_log_minimize_2x():
    s = _state(100.0, 50.0, "minimize")
    r = asyncio.run(log_improvement_reward([], s))
    assert abs(r - 1.0) < 0.01

def test_log_maximize_2x():
    s = _state(50.0, 100.0, "maximize")
    r = asyncio.run(log_improvement_reward([], s))
    assert abs(r - 1.0) < 0.01

def test_log_no_improvement():
    s = _state(100.0, 100.0, "minimize")
    r = asyncio.run(log_improvement_reward([], s))
    assert r == 0.0

def test_linear_improvement():
    s = _state(100.0, 80.0, "minimize")
    s["reward_normalization"] = 50.0
    r = asyncio.run(linear_improvement_reward([], s))
    assert abs(r - 0.4) < 0.01

def test_failure_penalty():
    s = _state(100.0, 50.0, failures=3)
    r = asyncio.run(failure_penalty([], s))
    assert abs(r - (-0.15)) < 0.01

def test_monotonic():
    history = [
        {"step": 0, "correct": True, "metric": 100},
        {"step": 1, "correct": True, "metric": 90},
        {"step": 2, "correct": True, "metric": 95},
        {"step": 3, "correct": True, "metric": 80},
    ]
    s = _state(100.0, 80.0, "minimize", history=history)
    r = asyncio.run(monotonic_improvement_reward([], s))
    # Steps 1,2,3 are non-baseline. Step 1: 90<100 improve. Step 2: 95>90 no. Step 3: 80<90 improve.
    # 2/3 = 0.667
    assert abs(r - 2 / 3) < 0.01

def test_build_rubric():
    rubric = build_rubric("log", weights={"improvement": 1.0, "monotonic": 0.25})
    assert rubric is not None
