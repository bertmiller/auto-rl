"""
KernelOptEnv: RL environment for training models to optimize
VLIW SIMD kernels by iterative code editing.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid

import verifiers as vf
from verifiers.types import Messages, State

from .rubric import rubric, BASELINE_CYCLES
from .sandbox import (
    KernelOptSandboxPool,
    sandbox_export_artifacts,
    sandbox_install_variant,
)
from .tools import edit_file, read_file, run_analysis, run_command, run_tests
from . import sandbox as sandbox_mod

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class KernelOptEnv(vf.StatefulToolEnv):
    """
    RL environment for training models to optimize kernels on a custom
    VLIW SIMD architecture.

    One episode = one optimization session. The agent iterates on
    perf_takehome.py for up to max_attempts optimization cycles,
    starting from the same naive baseline every time.

    Per-rollout state:
        sandbox_id: str               -- isolated working directory
        baseline_cycles: int           -- 147734 (naive scalar implementation)
        best_cycles: int               -- best cycle count this episode
        current_cycles: int            -- most recent cycle count
        optimization_history: list     -- [{step, cycles, speedup, correct, description}]
        num_failures: int              -- incorrect results or crashes
        optimization_step: int         -- current step counter
        episode_done: bool
    """

    def __init__(
        self,
        max_attempts: int = 20,
        sandbox_pool_size: int = 4,
        **kwargs,
    ):
        super().__init__(
            rubric=rubric,
            max_turns=max_attempts * 4,
            **kwargs,
        )
        self.max_attempts = max_attempts
        self.sandbox_pool = KernelOptSandboxPool(size=sandbox_pool_size)
        # Set module-level pool reference for sandbox operations
        sandbox_mod._pool = self.sandbox_pool

        # Register tools with sandbox_id hidden from model schema
        self.add_tool(read_file, args_to_skip=["sandbox_id"])
        self.add_tool(edit_file, args_to_skip=["sandbox_id"])
        self.add_tool(run_tests, args_to_skip=["sandbox_id"])
        self.add_tool(run_analysis, args_to_skip=["sandbox_id"])
        self.add_tool(run_command, args_to_skip=["sandbox_id"])

    async def setup_state(self, state: State, **kwargs) -> State:
        """
        Allocate a fresh sandbox, install the task's starting-point variant,
        and initialize episode state with the variant's baseline cycle count.
        """
        sandbox_id = await self.sandbox_pool.acquire()

        # Read task-specific starting point from dataset info
        task_info = state.get("info", {})
        baseline_cycles = task_info.get("baseline_cycles", BASELINE_CYCLES)
        variant_file = task_info.get("variant_file", "baseline_147734.py")

        # Install the variant's perf_takehome.py into the sandbox
        await sandbox_install_variant(sandbox_id, variant_file)

        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        state.update({
            "sandbox_id": sandbox_id,
            "episode_id": episode_id,
            "baseline_cycles": baseline_cycles,
            "best_cycles": baseline_cycles,
            "current_cycles": baseline_cycles,
            "variant_file": variant_file,
            "optimization_history": [{
                "step": 0,
                "cycles": baseline_cycles,
                "speedup": 1.0,
                "correct": True,
                "description": f"starting point ({variant_file})",
            }],
            "num_failures": 0,
            "optimization_step": 0,
            "episode_done": False,
        })

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
        Dispatch tool calls via parent, then post-process run_tests
        results to track optimization state.
        """
        # Collect tool_call_id -> tool_name mapping from the assistant message
        run_tests_call_ids: set[str] = set()
        last_msg = messages[-1] if messages else {}
        tool_calls = (
            last_msg.get("tool_calls", [])
            if isinstance(last_msg, dict)
            else getattr(last_msg, "tool_calls", []) or []
        )
        for tc in tool_calls:
            fn = (
                tc.get("function", {})
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            name = (
                fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            )
            tc_id = (
                tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            )
            if name == "run_tests":
                run_tests_call_ids.add(tc_id)

        tool_messages = await super().env_response(messages, state, **kwargs)

        # Post-process only run_tests tool responses
        for msg in tool_messages:
            tc_id = (
                msg.get("tool_call_id", "")
                if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            if tc_id not in run_tests_call_ids:
                continue

            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            if not content:
                continue

            if content.startswith("CORRECT"):
                m = re.search(r"cycles=(\d+)", content)
                if m:
                    cycles = int(m.group(1))
                    state["optimization_step"] = state.get("optimization_step", 0) + 1
                    step = state["optimization_step"]
                    state["current_cycles"] = cycles

                    if cycles < state["best_cycles"]:
                        state["best_cycles"] = cycles

                    baseline = state["baseline_cycles"]
                    state["optimization_history"].append({
                        "step": step,
                        "cycles": cycles,
                        "speedup": round(baseline / cycles, 2),
                        "correct": True,
                        "description": "",
                    })

                    if step >= self.max_attempts:
                        state["episode_done"] = True

            elif content.startswith("INCORRECT") or content.startswith("CRASH"):
                state["num_failures"] = state.get("num_failures", 0) + 1
                state["optimization_step"] = state.get("optimization_step", 0) + 1
                step = state["optimization_step"]

                state["optimization_history"].append({
                    "step": step,
                    "cycles": None,
                    "speedup": 0,
                    "correct": False,
                    "description": content[:200],
                })

                if step >= self.max_attempts:
                    state["episode_done"] = True

        return tool_messages

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    async def release_sandbox(self, state: State) -> None:
        sid = state.get("sandbox_id")
        if not sid:
            return

        episode_id = state.get("episode_id", sid)
        try:
            await sandbox_export_artifacts(sid, episode_id, dict(state))
        except Exception as e:
            log.warning("Failed to export artifacts for %s: %s", episode_id, e)

        self._log_episode_metrics(state)

        await self.sandbox_pool.release(sid)

    async def rollout(self, *args, **kwargs):
        """Override rollout to ensure sandbox cleanup after each episode."""
        completion, state = await super().rollout(*args, **kwargs)
        await self.release_sandbox(state)
        return completion, state

    def _log_episode_metrics(self, state: State) -> None:
        """Log per-episode metrics to wandb."""
        if wandb is None or wandb.run is None:
            return

        baseline = state.get("baseline_cycles", BASELINE_CYCLES)
        best = state.get("best_cycles", baseline)
        num_steps = state.get("optimization_step", 0)
        num_failures = state.get("num_failures", 0)
        history = state.get("optimization_history", [])
        correct_non_baseline = [
            e for e in history if e.get("correct") and e.get("step", 0) > 0
        ]

        best_so_far = baseline
        num_improvements = 0
        for e in correct_non_baseline:
            if e["cycles"] is not None and e["cycles"] < best_so_far:
                num_improvements += 1
                best_so_far = e["cycles"]

        speedup = baseline / max(best, 1)
        log2_speedup = math.log2(speedup) if speedup > 1 else 0.0

        episode_data = {
            "episode/baseline_cycles": baseline,
            "episode/best_cycles": best,
            "episode/speedup": round(speedup, 2),
            "episode/log2_speedup": round(log2_speedup, 3),
            "episode/num_attempts": num_steps,
            "episode/num_failures": num_failures,
            "episode/failure_rate": num_failures / max(num_steps, 1),
            "episode/num_improvements": num_improvements,
            "episode/monotonic_frac": num_improvements / max(len(correct_non_baseline), 1),
        }
        wandb.log(episode_data)

    async def shutdown_pool(self) -> None:
        await self.sandbox_pool.shutdown()
