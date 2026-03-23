"""
autoresearch-env: RL environment for karpathy/autoresearch optimization.

Exposes load_environment() as the prime-rl / verifiers entrypoint.
"""

from __future__ import annotations

from datasets import Dataset

from .env import AutoresearchEnv

SYSTEM_PROMPT = """\
You are an ML research agent. Your task is to improve the val_bpb
(validation bits per byte) of a small GPT training script by running
experiments autonomously.

Lower val_bpb is better. The training script runs for exactly 5 minutes
per experiment regardless of what you change.

Your tools:
  read_file(path)                      — read train.py, run.log, or results.tsv
  edit_file(path, content)             — rewrite train.py with your changes
  run_experiment()                     — run training and get val_bpb
  git_commit_or_revert(action, msg)    — keep or discard the change

The research loop:
  1. Read train.py to understand the current state.
  2. Form a hypothesis. Change one thing at a time.
  3. Edit train.py to implement it.
  4. Run the experiment.
  5. If val_bpb improved: commit. If not: revert.
  6. Repeat.

If a run crashes, read run.log to diagnose. Attempt one fix.
If the fix also crashes, revert and try a different idea.

You have 15 experiments. Use them to make meaningful, targeted progress.
"""

DATASET_REPEAT_N = 10_000


def _build_dataset() -> Dataset:
    task = {
        "task_id": "autoresearch-baseline",
        "prompt": SYSTEM_PROMPT,
        "answer": 0.975,  # Karpathy's best from ~90-experiment sessions
        "info": {
            "baseline_commit": "HEAD",
        },
    }
    return Dataset.from_list([task] * DATASET_REPEAT_N)


def load_environment(
    max_experiments: int = 15,
    sandbox_pool_size: int = 16,
    sandbox_image: str = "autoresearch-sandbox:latest",
    **kwargs,
) -> AutoresearchEnv:
    """
    Prime-rl / verifiers entrypoint.

    Returns a fully configured AutoresearchEnv ready for rollouts.
    """
    dataset = _build_dataset()

    return AutoresearchEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        max_experiments=max_experiments,
        sandbox_pool_size=sandbox_pool_size,
        sandbox_image=sandbox_image,
        **kwargs,
    )
