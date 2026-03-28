from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Sandbox:
    id: str
    path: Path


class SandboxPool:
    def __init__(
        self,
        size: int,
        challenge_dir: Path,
        sandboxes_dir: Path | None = None,
    ):
        self.size = size
        self.challenge_dir = Path(challenge_dir)
        self.sandboxes_dir = Path(sandboxes_dir or f"/tmp/auto-rl-{uuid.uuid4().hex[:8]}")
        self._available: asyncio.Queue[Sandbox] = asyncio.Queue()
        self._all: dict[str, Sandbox] = {}
        self._initialized = False

    def _copy_challenge(self, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.challenge_dir, dest)

    def _reset_sandbox(self, sandbox: Sandbox) -> None:
        if sandbox.path.exists():
            shutil.rmtree(sandbox.path)
        self._copy_challenge(sandbox.path)

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.sandboxes_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        for i in range(self.size):
            sid = f"sandbox-{i}"
            path = self.sandboxes_dir / sid
            await loop.run_in_executor(None, self._copy_challenge, path)
            sandbox = Sandbox(id=sid, path=path)
            self._all[sid] = sandbox
            await self._available.put(sandbox)
        self._initialized = True
        log.info("SandboxPool: %d sandboxes from %s", self.size, self.challenge_dir)

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

    def get(self, sandbox_id: str) -> Sandbox:
        return self._all[sandbox_id]

    async def shutdown(self) -> None:
        log.info("SandboxPool shut down")


async def sandbox_exec(
    pool: SandboxPool,
    sandbox_id: str,
    cmd: str,
    timeout: int = 60,
    env_extra: dict | None = None,
) -> tuple[int, str, str]:
    sandbox = pool.get(sandbox_id)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                cmd,
                cwd=str(sandbox.path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 1, "", f"Timed out after {timeout}s"
    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    err = stderr.decode("utf-8", errors="replace") if stderr else ""
    return proc.returncode or 0, out, err
