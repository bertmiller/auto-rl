from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib


@dataclass
class DatasetVariant:
    """A starting-point variant for curriculum training."""
    file: str          # filename in variants_dir
    baseline: float    # baseline metric for this variant


@dataclass
class ProblemConfig:
    name: str
    system_prompt: str
    challenge_dir: Path
    editable_files: list[str]
    readable_files: list[str]
    oracle_command: str
    metric_pattern: str
    metric_name: str
    metric_direction: str  # "minimize" or "maximize"
    oracle_timeout: int = 30
    reward_type: str = "log"  # "log" or "linear"
    reward_baseline: float | None = None
    reward_normalization: float = 1.0
    max_attempts: int = 20
    sandbox_pool_size: int = 4
    setup_hook: str | None = None
    reset_hook: str | None = None
    extra_tools: list[str] = field(default_factory=list)
    dataset_repeat: int = 10_000
    variants: list[DatasetVariant] = field(default_factory=list)
    variants_dir: Path | None = None
    variants_target: str | None = None  # defaults to first editable file
    extra: dict[str, Any] = field(default_factory=dict)

    def render_system_prompt(self, **kwargs) -> str:
        return self.system_prompt.format(**kwargs)

    def is_improvement(self, new: float, best: float) -> bool:
        if self.metric_direction == "minimize":
            return new < best
        return new > best

    def resolve_hook(self, hook_path: str) -> Callable:
        module_path, func_name = hook_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)


def load_config(toml_path: str | Path) -> ProblemConfig:
    toml_path = Path(toml_path)
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    prob = raw.get("problem", {})
    files = raw.get("files", {})
    oracle = raw.get("oracle", {})
    reward = raw.get("reward", {})
    episode = raw.get("episode", {})
    plugins = raw.get("plugins", {})
    ds = raw.get("dataset", {})

    challenge_dir = toml_path.parent / files.get("challenge_dir", "challenge")

    # Parse [dataset] variants
    variants = [
        DatasetVariant(file=v["file"], baseline=v["baseline"])
        for v in ds.get("variants", [])
    ]
    variants_dir = None
    if "variants_dir" in ds:
        variants_dir = toml_path.parent / ds["variants_dir"]

    return ProblemConfig(
        name=prob["name"],
        system_prompt=prob.get("system_prompt", ""),
        challenge_dir=challenge_dir,
        editable_files=files.get("editable", []),
        readable_files=files.get("readable", []),
        oracle_command=oracle["command"],
        metric_pattern=oracle["metric_pattern"],
        metric_name=oracle.get("metric_name", "score"),
        metric_direction=oracle.get("direction", "minimize"),
        oracle_timeout=oracle.get("timeout_secs", 30),
        reward_type=reward.get("type", "log"),
        reward_baseline=reward.get("baseline"),
        reward_normalization=reward.get("normalization", 1.0),
        max_attempts=episode.get("max_attempts", 20),
        sandbox_pool_size=episode.get("sandbox_pool_size", 4),
        setup_hook=plugins.get("setup_hook"),
        reset_hook=plugins.get("reset_hook"),
        extra_tools=plugins.get("extra_tools", []),
        dataset_repeat=episode.get("dataset_repeat", 10_000),
        variants=variants,
        variants_dir=variants_dir,
        variants_target=ds.get("target"),
        extra=raw,
    )
