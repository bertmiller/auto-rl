import asyncio
from pathlib import Path
from auto_rl.config import load_config
from auto_rl.sandbox import SandboxPool
from auto_rl.tools import create_tools

TOML = Path(__file__).parent / "fixtures" / "sample_problem" / "problem.toml"

def _setup():
    config = load_config(TOML)
    pool = SandboxPool(size=1, challenge_dir=config.challenge_dir)
    sid = asyncio.run(pool.acquire())
    tools = create_tools(config, pool)
    return config, pool, sid, tools

def test_read_file_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert "def solve" in result

def test_read_file_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["read_file"]("secret.py", sandbox_id=sid))
    assert "Error" in result

def test_edit_file_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["edit_file"]("solution.py", "x = 1", sandbox_id=sid))
    assert result == "OK"
    content = asyncio.run(tools["read_file"]("solution.py", sandbox_id=sid))
    assert content == "x = 1"

def test_edit_file_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["edit_file"]("verify.py", "hack", sandbox_id=sid))
    assert "Error" in result

def test_run_eval():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_eval"](sandbox_id=sid))
    assert "score=" in result

def test_run_command_allowed():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_command"]("ls", sandbox_id=sid))
    assert "solution.py" in result

def test_run_command_blocked():
    config, pool, sid, tools = _setup()
    result = asyncio.run(tools["run_command"]("rm -rf /", sandbox_id=sid))
    assert "Error" in result
