"""
Sandbox layer for autoresearch RL environment.

Each sandbox is an isolated directory clone of karpathy/autoresearch
with a dedicated GPU assigned via CUDA_VISIBLE_DEVICES. No Docker required.

Pre-provisioning:
  1. Clone karpathy/autoresearch to AUTORESEARCH_BASE_REPO (default /workspace/autoresearch)
  2. Run: cd $AUTORESEARCH_BASE_REPO && uv sync && uv run prepare.py
  Sandboxes are created as copies of this base repo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BASE_REPO = Path(os.environ.get("AUTORESEARCH_BASE_REPO", "/workspace/autoresearch"))
SANDBOXES_DIR = Path(os.environ.get("AUTORESEARCH_SANDBOXES_DIR", "/workspace/sandboxes"))
TRAIN_TIMEOUT = 330  # 5.5min hard cap (5min train + 30s buffer)


@dataclass
class Sandbox:
    id: str
    path: Path
    gpu_id: int


class AutoresearchSandboxPool:
    """
    Manages a pool of isolated directory sandboxes, each pinned to a GPU
    via CUDA_VISIBLE_DEVICES. Sandboxes are reused across episodes after reset.
    """

    def __init__(self, size: int = 2, gpu_ids: list[int] | None = None, **kwargs):
        self.size = size
        self.gpu_ids = gpu_ids if gpu_ids is not None else list(range(size))
        self._available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self._all: dict[str, Sandbox] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        SANDBOXES_DIR.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        for i in range(self.size):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            sandbox = await loop.run_in_executor(None, self._create_sandbox, i, gpu_id)
            self._all[sandbox.id] = sandbox
            await self._available.put(sandbox)
        self._initialized = True
        log.info("Sandbox pool initialized: %d sandboxes, GPUs %s", self.size, self.gpu_ids)

    def _create_sandbox(self, index: int, gpu_id: int) -> Sandbox:
        sid = f"sandbox-{index}"
        sandbox_path = SANDBOXES_DIR / sid
        if sandbox_path.exists():
            # Reset existing sandbox instead of re-cloning
            self._reset_sandbox_dir(sandbox_path)
        else:
            shutil.copytree(BASE_REPO, sandbox_path, symlinks=True)
        log.info("Created sandbox %s at %s on GPU %d", sid, sandbox_path, gpu_id)
        return Sandbox(id=sid, path=sandbox_path, gpu_id=gpu_id)

    async def acquire(self) -> str:
        if not self._initialized:
            await self.initialize()
        sandbox = await self._available.get()
        return sandbox.id

    async def release(self, sandbox_id: str) -> None:
        sandbox = self._all.get(sandbox_id)
        if sandbox is None:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._reset_sandbox_dir, sandbox.path)
        await self._available.put(sandbox)

    def _reset_sandbox_dir(self, path: Path) -> None:
        import subprocess
        subprocess.run(["git", "checkout", "main"], cwd=path, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=path, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=path, capture_output=True)
        # Delete any stale branches
        result = subprocess.run(
            ["git", "branch", "--list", "autoresearch/*"],
            cwd=path, capture_output=True, text=True,
        )
        for branch in result.stdout.strip().split("\n"):
            branch = branch.strip()
            if branch:
                subprocess.run(["git", "branch", "-D", branch], cwd=path, capture_output=True)

    async def shutdown(self) -> None:
        log.info("Sandbox pool shut down")


# --- Sandbox operations (called by tools) ---

_pool: AutoresearchSandboxPool | None = None


def _sandbox(sandbox_id: str) -> Sandbox:
    assert _pool is not None, "Sandbox pool not initialized"
    return _pool._all[sandbox_id]


async def _exec(sandbox_id: str, cmd: str, timeout: int = 60, env_extra: dict | None = None) -> tuple[int, str]:
    """Execute a shell command in the sandbox directory with GPU pinning."""
    sandbox = _sandbox(sandbox_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(sandbox.gpu_id)
    if env_extra:
        env.update(env_extra)

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                cwd=str(sandbox.path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            ),
            timeout=5,  # timeout for process creation only
        )
        stdout, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 1, f"Command timed out after {timeout}s: {cmd[:100]}"

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    return proc.returncode or 0, output


async def sandbox_read_file(sandbox_id: str, path: str) -> str:
    sandbox = _sandbox(sandbox_id)
    file_path = sandbox.path / path
    try:
        return file_path.read_text()
    except FileNotFoundError:
        return f"Error: {path} not found"
    except Exception as e:
        return f"Error reading {path}: {e}"


async def sandbox_write_file(sandbox_id: str, path: str, content: str) -> None:
    sandbox = _sandbox(sandbox_id)
    file_path = sandbox.path / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


async def sandbox_run_train(sandbox_id: str, train_secs: int = 300) -> dict:
    """Run train.py and return parsed results."""
    cmd = f"timeout {train_secs + 30} uv run train.py"
    t0 = time.monotonic()
    exit_code, output = await _exec(sandbox_id, cmd, timeout=TRAIN_TIMEOUT)
    elapsed = time.monotonic() - t0

    # Write run.log
    await sandbox_write_file(sandbox_id, "run.log", output)

    if exit_code != 0:
        lines = output.strip().split("\n")
        tail = "\n".join(lines[-50:])
        return {"crashed": True, "output": tail}

    # Parse val_bpb and peak_vram_mb from output
    val_bpb = None
    peak_vram = None
    for line in output.split("\n"):
        if m := re.match(r"^val_bpb:\s*([\d.]+)", line):
            val_bpb = float(m.group(1))
        if m := re.match(r"^peak_vram_mb:\s*(\d+)", line):
            peak_vram = int(m.group(1))

    if val_bpb is None:
        return {"crashed": True, "output": "No val_bpb found in output.\n" + output[-500:]}

    return {
        "crashed": False,
        "val_bpb": val_bpb,
        "peak_vram_mb": peak_vram or 0,
        "training_seconds": round(elapsed, 1),
    }


async def sandbox_git_checkout_branch(sandbox_id: str, branch: str) -> None:
    await _exec(sandbox_id, f"git checkout -b {branch}")


async def sandbox_git_commit(sandbox_id: str, message: str) -> str:
    """Commit train.py. Message written via file to avoid shell injection."""
    await sandbox_write_file(sandbox_id, ".commit_msg", message)
    await _exec(sandbox_id, "git add train.py")
    exit_code, output = await _exec(sandbox_id, "git commit -F .commit_msg")
    await _exec(sandbox_id, "rm -f .commit_msg")
    sha_match = re.search(r"\[[\w/]+ ([a-f0-9]+)\]", output)
    return sha_match.group(1) if sha_match else "unknown"


async def sandbox_git_revert(sandbox_id: str) -> None:
    await _exec(sandbox_id, "git checkout -- train.py")


ARTIFACTS_DIR = Path(os.environ.get("AUTORESEARCH_ARTIFACTS_DIR", "/workspace/artifacts"))


async def sandbox_export_artifacts(sandbox_id: str, episode_id: str, state: dict) -> Path:
    """
    Export episode artifacts from sandbox before it's reset.

    Exports: results.tsv, run.log, train.py, git.log, episode.json
    """
    sandbox = _sandbox(sandbox_id)
    episode_dir = ARTIFACTS_DIR / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["results.tsv", "run.log", "train.py"]:
        src = sandbox.path / filename
        if src.exists():
            shutil.copy2(src, episode_dir / filename)

    _, git_log = await _exec(sandbox_id, "git log --oneline --stat")
    (episode_dir / "git.log").write_text(git_log)

    history = state.get("experiment_history", [])
    summary = {
        "episode_id": episode_id,
        "sandbox_id": sandbox_id,
        "baseline_val_bpb": state.get("baseline_val_bpb"),
        "best_val_bpb": state.get("best_val_bpb"),
        "improvement": (state.get("baseline_val_bpb", 0) or 0) - (state.get("best_val_bpb", 0) or 0),
        "num_experiments": state.get("experiment_step", 0),
        "num_crashes": state.get("num_crashes", 0),
        "num_improvements": sum(1 for e in history if not e.get("crashed") and e.get("delta", 0) > 0 and e.get("step", 0) > 0),
        "experiment_history": history,
    }
    (episode_dir / "episode.json").write_text(json.dumps(summary, indent=2))

    log.info("Exported artifacts for episode %s to %s", episode_id, episode_dir)
    return episode_dir
