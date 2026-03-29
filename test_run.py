"""
Quick test run of the auto_rl framework against the sample problem.
No model server needed — manually drives the tool loop.
"""

import asyncio
import math
from pathlib import Path

from auto_rl import load_environment

TOML = Path("tests/fixtures/sample_problem/problem.toml")


async def run():
    print("Loading environment from TOML...")
    env = load_environment(str(TOML), sandbox_pool_size=1, max_attempts=5)
    print(f"  name:      {env.config.name}")
    print(f"  metric:    {env.config.metric_name} ({env.config.metric_direction})")
    print(f"  oracle:    {env.config.oracle_command}")
    print(f"  reward:    {env.config.reward_type}")
    print(f"  tools:     {list(env._tools.keys())}")
    print()

    # Setup episode state
    state = {"info": {"baseline_metric": 1.0}, "task": "sample", "answer": ""}
    state = await env.setup_state(state)
    sid = state["sandbox_id"]
    tools = env._tools
    print(f"Episode {state['episode_id']} started, sandbox={sid}")
    print(f"  baseline: {state['baseline_metric']}")
    print()

    import re

    def track_eval(result: str):
        """Simulate what env_response does: parse metric and update state."""
        metric_key = f"{env.config.metric_name}="
        if metric_key in result and "CRASH" not in result:
            m = re.search(rf"{re.escape(env.config.metric_name)}=([\d.eE+\-]+)", result)
            if m:
                val = float(m.group(1))
                state["optimization_step"] += 1
                state["current_metric"] = val
                if env.config.is_improvement(val, state["best_metric"]):
                    state["best_metric"] = val
                state["optimization_history"].append({
                    "step": state["optimization_step"],
                    "metric": val, "correct": True, "description": "",
                })
                if state["optimization_step"] >= env.config.max_attempts:
                    state["episode_done"] = True
        elif "CRASH" in result or "ERROR" in result:
            state["num_failures"] += 1
            state["optimization_step"] += 1
            state["optimization_history"].append({
                "step": state["optimization_step"],
                "metric": None, "correct": False, "description": result[:100],
            })
            if state["optimization_step"] >= env.config.max_attempts:
                state["episode_done"] = True

    # === Simulate agent loop ===

    # Step 1: Read current solution
    print("--- Step 1: Read solution.py ---")
    result = await tools["read_file"]("solution.py", sandbox_id=sid)
    print(f"  {result.strip()}")
    print()

    # Step 2: Read verify.py to understand scoring
    print("--- Step 2: Read verify.py ---")
    result = await tools["read_file"]("verify.py", sandbox_id=sid)
    print(f"  {result.strip()}")
    print()

    # Step 3: Run eval on baseline
    print("--- Step 3: Run eval (baseline) ---")
    result = await tools["run_eval"](sandbox_id=sid)
    print(f"  {result.split(' | ')[0]}")
    track_eval(result)
    print()

    # Step 4: Edit to improve score
    print("--- Step 4: Edit solution.py (return 10) ---")
    new_code = "def solve():\n    return 10.0\n"
    await tools["edit_file"]("solution.py", new_code, sandbox_id=sid)
    result = await tools["run_eval"](sandbox_id=sid)
    print(f"  run_eval -> {result.split(' | ')[0]}")
    track_eval(result)
    await tools["run_command"]("cp solution.py best.py", sandbox_id=sid)
    print(f"  checkpoint -> best={state['best_metric']}")
    print()

    # Step 5: Try another improvement
    print("--- Step 5: Edit (return 100) ---")
    new_code = "def solve():\n    return 100.0\n"
    await tools["edit_file"]("solution.py", new_code, sandbox_id=sid)
    result = await tools["run_eval"](sandbox_id=sid)
    print(f"  run_eval -> {result.split(' | ')[0]}")
    track_eval(result)
    await tools["run_command"]("cp solution.py best.py", sandbox_id=sid)
    print(f"  best={state['best_metric']}")
    print()

    # Step 6: Try a bad edit (crash)
    print("--- Step 6: Bad edit (crash) ---")
    bad_code = "def solve():\n    raise ValueError('oops')\n"
    await tools["edit_file"]("solution.py", bad_code, sandbox_id=sid)
    result = await tools["run_eval"](sandbox_id=sid)
    print(f"  run_eval -> {result[:60]}...")
    track_eval(result)
    await tools["run_command"]("cp best.py solution.py", sandbox_id=sid)
    print(f"  restored from checkpoint, failures={state['num_failures']}")
    print()

    # Step 7: One more improvement
    print("--- Step 7: Edit (return 1000) ---")
    new_code = "def solve():\n    return 1000.0\n"
    await tools["edit_file"]("solution.py", new_code, sandbox_id=sid)
    result = await tools["run_eval"](sandbox_id=sid)
    print(f"  run_eval -> {result.split(' | ')[0]}")
    track_eval(result)
    print()

    # === Summary ===
    baseline = state["baseline_metric"]
    best = state["best_metric"]
    steps = state["optimization_step"]
    failures = state["num_failures"]

    if env.config.metric_direction == "maximize" and best > baseline:
        ratio = best / baseline
    elif env.config.metric_direction == "minimize" and best < baseline:
        ratio = baseline / best
    else:
        ratio = 1.0
    log2_reward = math.log2(ratio) if ratio > 1 else 0.0

    print("=" * 50)
    print("EPISODE SUMMARY")
    print(f"  baseline:      {baseline}")
    print(f"  best:          {best}")
    print(f"  improvement:   {ratio:.1f}x")
    print(f"  log2 reward:   {log2_reward:.3f}")
    print(f"  steps:         {steps}")
    print(f"  failures:      {failures}")
    print(f"  episode_done:  {state['episode_done']}")
    print()
    print("Optimization history:")
    for h in state["optimization_history"]:
        status = "OK" if h["correct"] else "FAIL"
        metric = h["metric"] if h["metric"] is not None else "---"
        print(f"  step {h['step']}: {status}  metric={metric}")
    print("=" * 50)

    await env.sandbox_pool.release(sid)


if __name__ == "__main__":
    asyncio.run(run())
