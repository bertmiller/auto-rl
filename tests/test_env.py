import asyncio
from pathlib import Path

from datasets import Dataset

from auto_rl.config import load_config
from auto_rl.env import OptimizationEnv

TOML = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"


def _dummy_dataset():
    return Dataset.from_dict({"question": ["optimize"], "answer": [""]})


def test_env_creates():
    config = load_config(TOML)
    env = OptimizationEnv(config, dataset=_dummy_dataset())
    assert env.config.name == "sample-maximize"
    assert len(env.oai_tools) == 4  # read_file, edit_file, run_eval, run_command


def test_env_setup_state():
    config = load_config(TOML)
    env = OptimizationEnv(config, dataset=_dummy_dataset())
    state = {
        "info": {"baseline_metric": 1.0},
        "task": "sample",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    assert "sandbox_id" in state
    assert state["baseline_metric"] == 1.0
    assert state["best_metric"] == 1.0
    assert state["optimization_step"] == 0
    assert state["episode_done"] is False
    # Cleanup
    asyncio.run(env.sandbox_pool.release(state["sandbox_id"]))


def test_env_full_tool_loop():
    """Simulate an agent editing solution.py and running eval."""
    config = load_config(TOML)
    env = OptimizationEnv(config, dataset=_dummy_dataset())
    state = {"info": {"baseline_metric": 1.0}, "task": "sample", "answer": ""}
    state = asyncio.run(env.setup_state(state))

    sid = state["sandbox_id"]
    tools = env._tools

    # Read the solution
    result = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert "def solve" in result

    # Edit to return a higher score
    new_code = "def solve():\n    return 42.0\n"
    result = asyncio.run(tools["edit_file"]("solution.py", new_code, sandbox_id=sid))
    assert result == "OK"

    # Run eval
    result = asyncio.run(tools["run_eval"](sandbox_id=sid))
    assert "score=42.0" in result

    # Checkpoint
    result = asyncio.run(tools["run_command"]("cp solution.py best.py", sandbox_id=sid))

    # Cleanup
    asyncio.run(env.sandbox_pool.release(sid))


def test_load_environment():
    from auto_rl import load_environment
    env = load_environment(str(TOML))
    assert env.config.name == "sample-maximize"
    assert len(env.oai_tools) == 4
