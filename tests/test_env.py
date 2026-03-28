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
