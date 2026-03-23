"""
Sandbox layer for autoresearch RL environment.

Each sandbox is a Docker container with:
- karpathy/autoresearch cloned at HEAD
- uv sync completed, prepare.py run (TinyStories data + tokenizer ready)
- PyTorch + CUDA, no cold-start compilation cost
- A dedicated GPU for the duration of the episode
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.models.containers import Container

log = logging.getLogger(__name__)

DOCKER_IMAGE = "autoresearch-sandbox:latest"
REPO_DIR = "/workspace/autoresearch"
TRAIN_TIMEOUT = 330  # 5.5min hard cap (5min train + 30s buffer)


@dataclass
class Sandbox:
    id: str
    container: Container
    gpu_id: int


class AutoresearchSandboxPool:
    """
    Pre-provisions and manages a pool of Docker containers, each with a
    dedicated GPU. Containers are reused across episodes after cleanup.
    """

    def __init__(self, size: int = 16, image: str = DOCKER_IMAGE):
        self.size = size
        self.image = image
        self._client = docker.from_env()
        self._available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self._all: dict[str, Sandbox] = {}
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        loop = asyncio.get_event_loop()
        for i in range(self.size):
            sandbox = await loop.run_in_executor(None, self._create_sandbox, i)
            self._all[sandbox.id] = sandbox
            await self._available.put(sandbox)
        self._initialized = True
        log.info("Sandbox pool initialized: %d containers", self.size)

    def _create_sandbox(self, gpu_id: int) -> Sandbox:
        sid = f"autoresearch-{uuid.uuid4().hex[:8]}"
        container = self._client.containers.run(
            self.image,
            name=sid,
            detach=True,
            tty=True,
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=[str(gpu_id)],
                    capabilities=[["gpu"]],
                )
            ],
            working_dir=REPO_DIR,
            command="sleep infinity",
        )
        log.info("Created sandbox %s on GPU %d", sid, gpu_id)
        return Sandbox(id=sid, container=container, gpu_id=gpu_id)

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
        await loop.run_in_executor(None, self._reset_sandbox, sandbox)
        await self._available.put(sandbox)

    def _reset_sandbox(self, sandbox: Sandbox) -> None:
        c = sandbox.container
        c.exec_run(f"git -C {REPO_DIR} checkout main", demux=True)
        c.exec_run(f"git -C {REPO_DIR} clean -fd", demux=True)
        c.exec_run(f"git -C {REPO_DIR} reset --hard HEAD", demux=True)

    async def shutdown(self) -> None:
        loop = asyncio.get_event_loop()
        for sandbox in self._all.values():
            await loop.run_in_executor(None, sandbox.container.remove, True)
        self._all.clear()
        log.info("Sandbox pool shut down")


# --- Sandbox operations (called by tools) ---

# Module-level pool reference, set by AutoresearchEnv.__init__
_pool: AutoresearchSandboxPool | None = None


def _container(sandbox_id: str) -> Container:
    assert _pool is not None, "Sandbox pool not initialized"
    return _pool._all[sandbox_id].container


async def _exec(sandbox_id: str, cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Execute a command in the sandbox container with a timeout."""
    loop = asyncio.get_event_loop()
    c = _container(sandbox_id)
    try:
        exit_code, output = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: c.exec_run(cmd, demux=True, workdir=REPO_DIR),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return 1, f"Command timed out after {timeout}s: {cmd[:100]}"
    stdout = (output[0] or b"").decode("utf-8", errors="replace")
    stderr = (output[1] or b"").decode("utf-8", errors="replace")
    return exit_code, stdout + stderr


async def sandbox_read_file(sandbox_id: str, path: str) -> str:
    exit_code, content = await _exec(sandbox_id, f"cat {path}")
    if exit_code != 0:
        return f"Error reading {path}: {content}"
    return content


async def sandbox_write_file(sandbox_id: str, path: str, content: str) -> None:
    """Write file content into container via put_archive (no shell escaping)."""
    loop = asyncio.get_event_loop()
    c = _container(sandbox_id)

    data = content.encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=path)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    buf.seek(0)

    await loop.run_in_executor(
        None,
        lambda: c.put_archive(REPO_DIR, buf),
    )


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
    """Commit train.py with the given message. Message is written via file to avoid shell injection."""
    await sandbox_write_file(sandbox_id, ".commit_msg", message)
    await _exec(sandbox_id, "git add train.py")
    exit_code, output = await _exec(sandbox_id, "git commit -F .commit_msg")
    await _exec(sandbox_id, "rm -f .commit_msg")
    sha_match = re.search(r"\[[\w/]+ ([a-f0-9]+)\]", output)
    return sha_match.group(1) if sha_match else "unknown"


async def sandbox_git_revert(sandbox_id: str) -> None:
    await _exec(sandbox_id, "git checkout -- train.py")


ARTIFACTS_DIR = Path(os.environ.get("AUTORESEARCH_ARTIFACTS_DIR", "/tmp/autoresearch-artifacts"))


async def sandbox_export_artifacts(sandbox_id: str, episode_id: str, state: dict) -> Path:
    """
    Export episode artifacts from sandbox before it's reset.
    Returns path to the episode artifact directory.

    Exports:
    - results.tsv (experiment history)
    - run.log (last training output)
    - train.py (final state)
    - git.log (full commit history for this episode branch)
    - episode.json (structured episode summary for wandb)
    """
    episode_dir = ARTIFACTS_DIR / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    c = _container(sandbox_id)

    # Export files via get_archive
    for filename in ["results.tsv", "run.log", "train.py"]:
        try:
            bits, _ = await loop.run_in_executor(
                None,
                lambda f=filename: c.get_archive(f"{REPO_DIR}/{f}"),
            )
            tar_buf = io.BytesIO()
            for chunk in bits:
                tar_buf.write(chunk)
            tar_buf.seek(0)
            with tarfile.open(fileobj=tar_buf, mode="r") as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f:
                    (episode_dir / filename).write_bytes(f.read())
        except Exception as e:
            log.warning("Failed to export %s from %s: %s", filename, sandbox_id, e)

    # Export git log
    _, git_log = await _exec(sandbox_id, "git log --oneline --stat")
    (episode_dir / "git.log").write_text(git_log)

    # Write structured episode summary
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
