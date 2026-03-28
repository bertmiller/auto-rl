"""
autoresearch-env: RL environment for karpathy/autoresearch optimization.

Exposes load_environment() as the prime-rl / verifiers entrypoint.
"""

from __future__ import annotations

from datasets import Dataset

from .env import AutoresearchEnv

SYSTEM_PROMPT_TEMPLATE = """\
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

You have {max_experiments} experiments. Use them to make meaningful, targeted progress.
"""

DATASET_REPEAT_N = 10_000


def _build_dataset(system_prompt: str) -> Dataset:
    task = {
        "task_id": "autoresearch-baseline",
        "prompt": system_prompt,
        "answer": 0.975,  # Karpathy's best from ~90-experiment sessions
        "info": {
            "baseline_commit": "HEAD",
        },
    }
    return Dataset.from_list([task] * DATASET_REPEAT_N)


def load_environment(
    max_experiments: int = 15,
    sandbox_pool_size: int = 2,
    sandbox_gpu_ids: list[int] | None = None,
    **kwargs,
) -> AutoresearchEnv:
    """
    Prime-rl / verifiers entrypoint.

    Args:
        sandbox_gpu_ids: Explicit list of GPU device IDs for sandbox workers.
                         E.g. [2, 3] to reserve GPUs 0-1 for inference/trainer.
                         Defaults to [0, 1, ..., sandbox_pool_size-1].
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_experiments=max_experiments)
    dataset = _build_dataset(system_prompt)

    return AutoresearchEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        max_experiments=max_experiments,
        sandbox_pool_size=sandbox_pool_size,
        sandbox_gpu_ids=sandbox_gpu_ids,
        **kwargs,
    )
