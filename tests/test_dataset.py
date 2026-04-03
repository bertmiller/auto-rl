import asyncio
from pathlib import Path

from datasets import Dataset

from auto_rl.config import DatasetVariant, load_config
from auto_rl.env import OptimizationEnv
from auto_rl import _build_dataset, load_environment

VARIANT_TOML = Path(__file__).parent / "fixtures" / "variant_problem" / "problem.toml"
PLAIN_TOML = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"


# --- Config parsing ---

def test_config_parses_variants():
    config = load_config(VARIANT_TOML)
    assert len(config.variants) == 3
    assert config.variants[0].file == "baseline.py"
    assert config.variants[0].baseline == 1.0
    assert config.variants[2].file == "variant_10x.py"
    assert config.variants[2].baseline == 10.0


def test_config_parses_variants_dir():
    config = load_config(VARIANT_TOML)
    assert config.variants_dir is not None
    assert config.variants_dir.name == "starting_points"
    assert config.variants_dir.exists()


def test_config_parses_target():
    config = load_config(VARIANT_TOML)
    assert config.variants_target == "solution.py"


def test_config_no_dataset_section():
    config = load_config(PLAIN_TOML)
    assert config.variants == []
    assert config.variants_dir is None
    assert config.variants_target is None


# --- Dataset construction ---

def test_variant_dataset_has_all_variants():
    config = load_config(VARIANT_TOML)
    dataset, _ = _build_dataset(config)
    # 3 variants cycling over 10,000 rows
    assert len(dataset) == config.dataset_repeat
    # First 3 rows should be the 3 variants
    assert dataset[0]["info"]["variant_file"] == "baseline.py"
    assert dataset[1]["info"]["variant_file"] == "variant_5x.py"
    assert dataset[2]["info"]["variant_file"] == "variant_10x.py"


def test_variant_dataset_baselines():
    config = load_config(VARIANT_TOML)
    dataset, _ = _build_dataset(config)
    assert dataset[0]["info"]["baseline_metric"] == 1.0
    assert dataset[1]["info"]["baseline_metric"] == 5.0
    assert dataset[2]["info"]["baseline_metric"] == 10.0


def test_variant_dataset_prompts_contain_baseline():
    config = load_config(VARIANT_TOML)
    dataset, _ = _build_dataset(config)
    assert "1.0" in dataset[0]["prompt"]
    assert "5.0" in dataset[1]["prompt"]
    assert "10.0" in dataset[2]["prompt"]


def test_plain_dataset_no_variant_file():
    config = load_config(PLAIN_TOML)
    dataset, _ = _build_dataset(config)
    assert "variant_file" not in dataset[0]["info"]


# --- Variant installation ---

def test_variant_installed_in_sandbox():
    config = load_config(VARIANT_TOML)
    env = OptimizationEnv(config, dataset=Dataset.from_dict({"question": ["x"], "answer": [""]}))

    # Use the variant_5x info
    state = {
        "info": {"baseline_metric": 5.0, "variant_file": "variant_5x.py"},
        "task": "test",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    sid = state["sandbox_id"]

    # Read what's in solution.py — should be the variant_5x content
    sandbox = env.sandbox_pool.get(sid)
    content = (sandbox.path / "solution.py").read_text()
    assert "5.0" in content

    assert state["baseline_metric"] == 5.0
    asyncio.run(env.sandbox_pool.release(sid))


def test_variant_10x_installed():
    config = load_config(VARIANT_TOML)
    env = OptimizationEnv(config, dataset=Dataset.from_dict({"question": ["x"], "answer": [""]}))

    state = {
        "info": {"baseline_metric": 10.0, "variant_file": "variant_10x.py"},
        "task": "test",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    sid = state["sandbox_id"]

    sandbox = env.sandbox_pool.get(sid)
    content = (sandbox.path / "solution.py").read_text()
    assert "10.0" in content
    asyncio.run(env.sandbox_pool.release(sid))


def test_no_variant_leaves_original():
    """When info has no variant_file, original challenge file is untouched."""
    config = load_config(VARIANT_TOML)
    env = OptimizationEnv(config, dataset=Dataset.from_dict({"question": ["x"], "answer": [""]}))

    state = {
        "info": {"baseline_metric": 1.0},
        "task": "test",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    sid = state["sandbox_id"]

    sandbox = env.sandbox_pool.get(sid)
    content = (sandbox.path / "solution.py").read_text()
    assert "1.0" in content  # original challenge/solution.py returns 1.0
    asyncio.run(env.sandbox_pool.release(sid))


# --- Full tool loop with variant ---

def test_variant_eval_returns_variant_score():
    """run_eval on a variant returns the variant's score."""
    config = load_config(VARIANT_TOML)
    env = OptimizationEnv(config, dataset=Dataset.from_dict({"question": ["x"], "answer": [""]}))

    state = {
        "info": {"baseline_metric": 5.0, "variant_file": "variant_5x.py"},
        "task": "test",
        "answer": "",
    }
    state = asyncio.run(env.setup_state(state))
    sid = state["sandbox_id"]

    result = asyncio.run(env._tools["run_eval"](sandbox_id=sid))
    assert "score=5.0" in result
    asyncio.run(env.sandbox_pool.release(sid))


# --- load_environment integration ---

def test_load_environment_with_variants():
    env = load_environment(str(VARIANT_TOML))
    assert len(env.config.variants) == 3
    assert env.config.name == "sample-variants"
