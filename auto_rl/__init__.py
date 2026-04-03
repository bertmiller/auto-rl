"""
auto-rl: Generic RL environments for optimization-by-code-editing.

Usage:
    from auto_rl import load_environment
    env = load_environment("path/to/problem.toml")
"""

from .config import DatasetVariant, ProblemConfig, load_config
from .env import OptimizationEnv
from .rewards import build_rubric
from .sandbox import SandboxPool

from datasets import Dataset


def _build_dataset(config: ProblemConfig) -> tuple[Dataset, str]:
    """Build dataset and system prompt from config.

    When variants are defined, creates one row per variant (each with its own
    baseline in ``info``), then cycles the list to fill ``dataset_repeat``.
    Otherwise falls back to a single repeated row.

    Returns:
        (dataset, system_prompt) — system_prompt is a representative rendering
        used as the env-level default (individual rows carry their own prompts).
    """
    if config.variants:
        tasks: list[dict] = []
        for v in config.variants:
            prompt = config.render_system_prompt(
                max_attempts=config.max_attempts,
                metric_name=config.metric_name,
                baseline_metric=v.baseline,
            )
            tasks.append({
                "task_id": f"{config.name}-{v.file}",
                "prompt": prompt,
                "answer": str(v.baseline),
                "info": {
                    "baseline_metric": v.baseline,
                    "variant_file": v.file,
                },
            })
        full = tasks * (config.dataset_repeat // len(tasks) + 1)
        dataset = Dataset.from_list(full[: config.dataset_repeat])
        # Use the first variant's baseline for the env-level prompt
        system_prompt = tasks[0]["prompt"]
    else:
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

    return dataset, system_prompt


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

    dataset, system_prompt = _build_dataset(config)

    return OptimizationEnv(
        config=config,
        dataset=dataset,
        system_prompt=system_prompt,
    )
