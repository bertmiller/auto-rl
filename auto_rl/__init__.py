"""
auto-rl: Generic RL environments for optimization-by-code-editing.

Usage:
    from auto_rl import load_environment
    env = load_environment("path/to/problem.toml")
"""

from .config import ProblemConfig, load_config
from .env import OptimizationEnv
from .rewards import build_rubric
from .sandbox import SandboxPool

from datasets import Dataset


def load_environment(config_path: str, **overrides) -> OptimizationEnv:
    """
    Load an optimization RL environment from a TOML config file.

    Args:
        config_path: Path to problem.toml
        **overrides: Override config fields (max_attempts, sandbox_pool_size, etc.)

    Returns:
        Configured OptimizationEnv ready for training.
    """
    config = load_config(config_path)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Build dataset
    baseline = config.reward_baseline or 0.0
    system_prompt = config.render_system_prompt(
        max_attempts=config.max_attempts,
        metric_name=config.metric_name,
        baseline_metric=baseline,
    )
    task = {
        "task_id": f"{config.name}-baseline",
        "prompt": system_prompt,
        "answer": str(baseline),
        "info": {"baseline_metric": baseline},
    }
    dataset = Dataset.from_list([task] * config.dataset_repeat)

    return OptimizationEnv(
        config=config,
        dataset=dataset,
        system_prompt=system_prompt,
    )
