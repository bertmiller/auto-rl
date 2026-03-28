"""
AutoresearchEnv: RL environment for training models to improve
karpathy/autoresearch val_bpb through iterative code editing.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

import verifiers as vf
from verifiers.types import Messages, State

from .rubric import rubric
from .sandbox import (
    AutoresearchSandboxPool,
    sandbox_export_artifacts,
    sandbox_git_checkout_branch,
    sandbox_run_train,
    sandbox_write_file,
)
from .tools import edit_file, git_commit_or_revert, read_file, run_experiment
from . import sandbox as sandbox_mod

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class AutoresearchEnv(vf.StatefulToolEnv):
    """
    RL environment for training models to do better at karpathy/autoresearch.

    One episode = one research session. The agent iterates on train.py
    for up to max_experiments cycles, starting from the same baseline
    every time.

    Per-rollout state:
        sandbox_id: str              -- isolated container with dedicated GPU
        baseline_val_bpb: float      -- score from run-0 before any edits
        best_val_bpb: float          -- best score seen this episode
        experiment_history: list     -- [{step, val_bpb, delta, crashed, message}]
        num_crashes: int
        experiment_step: int
        episode_done: bool
    """

    def __init__(
        self,
        max_experiments: int = 15,
        sandbox_pool_size: int = 2,
        sandbox_gpu_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(
            rubric=rubric,
            max_turns=max_experiments * 4,
            **kwargs,
        )
        self.max_experiments = max_experiments
        self.sandbox_pool = AutoresearchSandboxPool(
            size=sandbox_pool_size,
            gpu_ids=sandbox_gpu_ids,
        )
        # Set module-level pool reference for sandbox operations
        sandbox_mod._pool = self.sandbox_pool

        # Register tools with sandbox_id hidden from model schema
        self.add_tool(read_file, args_to_skip=["sandbox_id"])
        self.add_tool(edit_file, args_to_skip=["sandbox_id"])
        self.add_tool(run_experiment, args_to_skip=["sandbox_id"])
        self.add_tool(git_commit_or_revert, args_to_skip=["sandbox_id"])

    async def setup_state(self, state: State, **kwargs) -> State:
        """
        Allocate a fresh sandbox, create an isolated branch,
        and establish the baseline val_bpb by running train.py once
        before the model receives any context.
        """
        sandbox_id = await self.sandbox_pool.acquire()

        # Isolate this rollout on its own branch
        episode_id = uuid.uuid4().hex[:8]
        state["episode_id"] = f"ep-{episode_id}"
        branch = f"autoresearch/rl-episode-{episode_id}"
        await sandbox_git_checkout_branch(sandbox_id, branch)

        # Establish baseline (run-0)
        baseline_result = await sandbox_run_train(sandbox_id, train_secs=300)

        if baseline_result["crashed"]:
            state.update({
                "sandbox_id": sandbox_id,
                "baseline_val_bpb": 1.0,
                "best_val_bpb": 1.0,
                "experiment_history": [],
                "num_crashes": 1,
                "experiment_step": 0,
                "episode_done": True,
            })
            return await super().setup_state(state, **kwargs)

        baseline_val_bpb = baseline_result["val_bpb"]

        state.update({
            "sandbox_id": sandbox_id,
            "baseline_val_bpb": baseline_val_bpb,
            "best_val_bpb": baseline_val_bpb,
            "experiment_history": [{
                "step": 0,
                "val_bpb": baseline_val_bpb,
                "delta": 0.0,
                "crashed": False,
                "message": "baseline",
            }],
            "num_crashes": 0,
            "experiment_step": 0,
            "episode_done": False,
        })

        await self._write_results_tsv(sandbox_id, state["experiment_history"])

        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages, state: State, **kwargs
    ) -> dict:
        """Inject sandbox_id into all tool calls."""
        updated = dict(tool_args)
        updated["sandbox_id"] = state["sandbox_id"]
        return updated

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Dispatch tool calls via parent, then post-process run_experiment
        results to track experiment state.

        We identify run_experiment results by matching tool_call_ids from
        the assistant's tool_calls back to the tool name, rather than
        pattern-matching on response content (which would false-positive
        on read_file("run.log") etc.).
        """
        # Collect tool_call_id -> tool_name mapping from the assistant message
        run_experiment_call_ids: set[str] = set()
        last_msg = messages[-1] if messages else {}
        tool_calls = last_msg.get("tool_calls", []) if isinstance(last_msg, dict) else getattr(last_msg, "tool_calls", []) or []
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            if name == "run_experiment":
                run_experiment_call_ids.add(tc_id)

        tool_messages = await super().env_response(messages, state, **kwargs)

        # Post-process only run_experiment tool responses
        for msg in tool_messages:
            tc_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
            if tc_id not in run_experiment_call_ids:
                continue

            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if not content:
                continue

            if "val_bpb=" in content and "CRASH" not in content:
                m = re.search(r"val_bpb=([\d.]+)", content)
                if m:
                    val_bpb = float(m.group(1))
                    state["experiment_step"] = state.get("experiment_step", 0) + 1
                    step = state["experiment_step"]
                    delta = state["best_val_bpb"] - val_bpb

                    if val_bpb < state["best_val_bpb"]:
                        state["best_val_bpb"] = val_bpb

                    state["experiment_history"].append({
                        "step": step,
                        "val_bpb": val_bpb,
                        "delta": delta,
                        "crashed": False,
                        "message": "",
                    })

                    if step >= self.max_experiments:
                        state["episode_done"] = True

                    await self._write_results_tsv(
                        state["sandbox_id"], state["experiment_history"]
                    )

            elif "CRASH" in content:
                state["num_crashes"] = state.get("num_crashes", 0) + 1
                state["experiment_step"] = state.get("experiment_step", 0) + 1
                step = state["experiment_step"]

                state["experiment_history"].append({
                    "step": step,
                    "val_bpb": None,
                    "delta": 0.0,
                    "crashed": True,
                    "message": content[:200],
                })

                if step >= self.max_experiments:
                    state["episode_done"] = True

                await self._write_results_tsv(
                    state["sandbox_id"], state["experiment_history"]
                )

        return tool_messages

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    @vf.cleanup
    async def release_sandbox(self, state: State) -> None:
        sid = state.get("sandbox_id")
        if not sid:
            return

        # Export artifacts before resetting the sandbox
        episode_id = state.get("episode_id", sid)
        try:
            await sandbox_export_artifacts(sid, episode_id, dict(state))
        except Exception as e:
            log.warning("Failed to export artifacts for %s: %s", episode_id, e)

        # Log structured episode metrics to wandb
        self._log_episode_metrics(state)

        await self.sandbox_pool.release(sid)

    def _log_episode_metrics(self, state: State) -> None:
        """Log per-episode metrics to wandb as a summary row."""
        if wandb is None or wandb.run is None:
            return

        history = state.get("experiment_history", [])
        baseline = state.get("baseline_val_bpb")
        best = state.get("best_val_bpb")
        num_exp = state.get("experiment_step", 0)
        num_crashes = state.get("num_crashes", 0)
        non_crash = [e for e in history if not e.get("crashed") and e.get("step", 0) > 0]
        num_improvements = sum(1 for e in non_crash if e.get("delta", 0) > 0)

        episode_data = {
            "episode/baseline_val_bpb": baseline,
            "episode/best_val_bpb": best,
            "episode/improvement": (baseline or 0) - (best or 0),
            "episode/num_experiments": num_exp,
            "episode/num_crashes": num_crashes,
            "episode/crash_rate": num_crashes / max(num_exp, 1),
            "episode/num_improvements": num_improvements,
            "episode/monotonic_frac": num_improvements / max(len(non_crash), 1),
        }
        wandb.log(episode_data)

        # Log experiment trajectory as a wandb table
        columns = ["step", "val_bpb", "delta", "crashed"]
        table = wandb.Table(columns=columns)
        for e in history:
            table.add_data(
                e["step"],
                e["val_bpb"] if e["val_bpb"] is not None else None,
                e["delta"],
                e["crashed"],
            )
        wandb.log({"episode/trajectory": table})

    @vf.teardown
    async def shutdown_pool(self) -> None:
        await self.sandbox_pool.shutdown()

    async def _write_results_tsv(self, sandbox_id: str, history: list[dict]) -> None:
        """Write experiment history as TSV for the agent to read."""
        lines = ["step\tval_bpb\tdelta\tcrashed\tmessage"]
        for e in history:
            val = f"{e['val_bpb']:.4f}" if e["val_bpb"] is not None else "CRASH"
            lines.append(
                f"{e['step']}\t{val}\t{e['delta']:.4f}\t{e['crashed']}\t{e['message']}"
            )
        await sandbox_write_file(sandbox_id, "results.tsv", "\n".join(lines) + "\n")
