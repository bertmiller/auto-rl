import asyncio
from pathlib import Path
from auto_rl.sandbox import SandboxPool, sandbox_exec

CHALLENGE = Path(__file__).parent / "fixtures" / "sample_problem" / "challenge"

def test_pool_init_and_acquire():
    pool = SandboxPool(size=2, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    assert sid.startswith("sandbox-")
    sandbox = pool.get(sid)
    assert (sandbox.path / "solution.py").exists()
    assert (sandbox.path / "verify.py").exists()

def test_pool_release_resets():
    pool = SandboxPool(size=1, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    sandbox = pool.get(sid)
    (sandbox.path / "solution.py").write_text("modified")
    (sandbox.path / "junk.py").write_text("junk")
    asyncio.run(pool.release(sid))
    sid2 = asyncio.run(pool.acquire())
    sandbox2 = pool.get(sid2)
    assert sandbox2.path == sandbox.path
    assert "modified" not in (sandbox2.path / "solution.py").read_text()
    assert not (sandbox2.path / "junk.py").exists()

def test_exec():
    pool = SandboxPool(size=1, challenge_dir=CHALLENGE)
    sid = asyncio.run(pool.acquire())
    code, stdout, stderr = asyncio.run(sandbox_exec(pool, sid, "echo hello"))
    assert code == 0
    assert "hello" in stdout
