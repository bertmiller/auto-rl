from __future__ import annotations

import logging
import re
import uuid

import verifiers as vf
from verifiers.types import Messages, State

from .config import ProblemConfig
from .rewards import build_rubric
from .sandbox import SandboxPool
from .tools import create_tools

log = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class OptimizationEnv(vf.StatefulToolEnv):
    """
    Generic RL environment for optimization-by-code-editing.
    Loads behavior from a ProblemConfig.
    """

    def __init__(self, config: ProblemConfig, **kwargs):
        self.config = config
        rubric = build_rubric(
            reward_type=config.reward_type,
            weights=kwargs.pop("reward_weights", None),
        )
        super().__init__(
            rubric=rubric,
            max_turns=config.max_attempts * 4,
            **kwargs,
        )
        self.sandbox_pool = SandboxPool(
            size=config.sandbox_pool_size,
            challenge_dir=config.challenge_dir,
        )
        self._tools = create_tools(config, self.sandbox_pool)

        for tool_fn in self._tools.values():
            self.add_tool(tool_fn, args_to_skip=["sandbox_id"])

        # Register plugin tools
        for hook_path in config.extra_tools:
            fn = config.resolve_hook(hook_path)
            self.add_tool(fn, args_to_skip=["sandbox_id"])

    async def setup_state(self, state: State, **kwargs) -> State:
        sandbox_id = await self.sandbox_pool.acquire()

        task_info = state.get("info", {})
        baseline = task_info.get("baseline_metric", self.config.reward_baseline or 0.0)

        # Run setup hook if configured
        if self.config.setup_hook:
            hook = self.config.resolve_hook(self.config.setup_hook)
            await hook(sandbox_id=sandbox_id, state=state, pool=self.sandbox_pool, config=self.config)

        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        state.update({
            "sandbox_id": sandbox_id,
            "episode_id": episode_id,
            "baseline_metric": baseline,
            "best_metric": baseline,
            "current_metric": baseline,
            "metric_direction": self.config.metric_direction,
            "metric_name": self.config.metric_name,
            "reward_normalization": self.config.reward_normalization,
            "optimization_history": [{
                "step": 0,
                "metric": baseline,
                "correct": True,
                "description": "baseline",
            }],
            "num_failures": 0,
            "optimization_step": 0,
            "episode_done": False,
        })
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages, state: State, **kwargs
    ) -> dict:
        updated = dict(tool_args)
        updated["sandbox_id"] = state["sandbox_id"]
        return updated

    async def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        # Find run_eval call IDs
        eval_call_ids: set[str] = set()
        last_msg = messages[-1] if messages else {}
        tool_calls = (
            last_msg.get("tool_calls", [])
            if isinstance(last_msg, dict)
            else getattr(last_msg, "tool_calls", []) or []
        )
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            if name == "run_eval":
                eval_call_ids.add(tc_id)

        tool_messages, state = await super().env_response(messages, state, **kwargs)

        for msg in tool_messages:
            tc_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
            if tc_id not in eval_call_ids:
                continue

            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if not content:
                continue

            metric_key = f"{self.config.metric_name}="
            if metric_key in content and "CRASH" not in content:
                m = re.search(rf"{re.escape(self.config.metric_name)}=([\d.eE+\-]+)", content)
                if m:
                    metric_val = float(m.group(1))
                    state["optimization_step"] = state.get("optimization_step", 0) + 1
                    step = state["optimization_step"]
                    state["current_metric"] = metric_val

                    if self.config.is_improvement(metric_val, state["best_metric"]):
                        state["best_metric"] = metric_val

                    state["optimization_history"].append({
                        "step": step,
                        "metric": metric_val,
                        "correct": True,
                        "description": "",
                    })
                    if step >= self.config.max_attempts:
                        state["episode_done"] = True

            elif "CRASH" in content or "ERROR" in content:
                state["num_failures"] = state.get("num_failures", 0) + 1
                state["optimization_step"] = state.get("optimization_step", 0) + 1
                step = state["optimization_step"]
                state["optimization_history"].append({
                    "step": step,
                    "metric": None,
                    "correct": False,
                    "description": content[:200],
                })
                if step >= self.config.max_attempts:
                    state["episode_done"] = True

        return tool_messages, state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        return state.get("episode_done", False)

    async def rollout(self, *args, **kwargs) -> tuple[Messages, State]:
        completion, state = await super().rollout(*args, **kwargs)
        await self._release_sandbox(state)
        return completion, state

    async def _release_sandbox(self, state: State) -> None:
        sid = state.get("sandbox_id")
        if not sid:
            return
        self._log_episode_metrics(state)
        await self.sandbox_pool.release(sid)

    def _log_episode_metrics(self, state: State) -> None:
        if wandb is None or wandb.run is None:
            return
        baseline = state.get("baseline_metric", 0)
        best = state.get("best_metric", baseline)
        steps = state.get("optimization_step", 0)
        failures = state.get("num_failures", 0)
        wandb.log({
            "episode/baseline": baseline,
            "episode/best": best,
            "episode/steps": steps,
            "episode/failures": failures,
            "episode/failure_rate": failures / max(steps, 1),
        })

    async def shutdown_pool(self) -> None:
        await self.sandbox_pool.shutdown()
