"""
Smoke test: verify the kernel-opt environment works after repo restructure.
Exercises sandbox creation, tool dispatch, and oracle execution.
"""

import asyncio
import re
import sys

sys.path.insert(0, ".")
from kernel_opt_env.sandbox import (
    KernelOptSandboxPool,
    sandbox_read_file,
    sandbox_run_tests,
    sandbox_run_analysis,
    sandbox_run_command,
    sandbox_write_file,
    sandbox_install_variant,
)
from kernel_opt_env import sandbox as sandbox_mod


async def smoke_test():
    print("=== Kernel Opt Environment Smoke Test ===\n")

    # 1. Initialize sandbox pool
    print("1. Initializing sandbox pool...")
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()
    sid = await pool.acquire()
    print(f"   Sandbox acquired: {sid}")
    sandbox = pool._all[sid]
    print(f"   Path: {sandbox.path}")
    assert sandbox.path.exists(), "Sandbox directory missing"
    print("   OK\n")

    # 2. Read challenge files
    print("2. Reading challenge files...")
    perf = await sandbox_read_file(sid, "perf_takehome.py")
    assert "KernelBuilder" in perf, "perf_takehome.py missing KernelBuilder"
    print(f"   perf_takehome.py: {len(perf)} chars, has KernelBuilder")

    problem = await sandbox_read_file(sid, "problem.py")
    assert "Machine" in problem, "problem.py missing Machine class"
    print(f"   problem.py: {len(problem)} chars, has Machine")
    print("   OK\n")

    # 3. Run baseline tests
    print("3. Running baseline tests (this takes a few seconds)...")
    result = await sandbox_run_tests(sid)
    print(f"   crashed: {result['crashed']}")
    print(f"   correct: {result.get('correct', 'N/A')}")
    if not result["crashed"]:
        print(f"   cycles: {result.get('cycles', 'N/A')}")
        print(f"   speedup: {result.get('speedup', 'N/A')}x")
        assert result["correct"], f"Baseline should be correct but got: {result}"
        assert result["cycles"] == 147734, f"Expected 147734 cycles, got {result['cycles']}"
    else:
        print(f"   ERROR: baseline crashed!\n   {result['output'][:300]}")
        return False
    print("   OK\n")

    # 4. Run analysis script
    print("4. Running analysis script...")
    script = """\
from perf_takehome import KernelBuilder
kb = KernelBuilder()
kb.build_kernel(10, 1023, 256, 16)
print(f"Total instructions: {len(kb.instrs)}")
"""
    result = await sandbox_run_analysis(sid, script, timeout_secs=30)
    assert result["success"], f"Analysis failed: {result['output']}"
    print(f"   {result['output'].strip()}")
    print("   OK\n")

    # 5. Run shell command
    print("5. Running shell commands...")
    result = await sandbox_run_command(sid, "ls", timeout_secs=5)
    assert result["success"], f"ls failed: {result['output']}"
    print(f"   Files: {result['output'].strip()}")

    result = await sandbox_run_command(sid, "cp perf_takehome.py best.py", timeout_secs=5)
    assert result["success"], "cp failed"
    print("   Checkpoint: cp perf_takehome.py best.py -> OK")
    print("   OK\n")

    # 6. Test starting point variant
    print("6. Installing starting-point variant...")
    await sandbox_install_variant(sid, "variant_15973.py")
    result = await sandbox_run_tests(sid)
    assert not result["crashed"], f"Variant crashed: {result['output'][:200]}"
    assert result["correct"], f"Variant incorrect"
    print(f"   variant_15973.py: {result['cycles']} cycles ({result['speedup']}x)")
    assert result["cycles"] < 20000, f"Expected ~15973 cycles, got {result['cycles']}"
    print("   OK\n")

    # 7. Cleanup
    await pool.release(sid)
    print("7. Sandbox released and reset\n")

    print("=== ALL CHECKS PASSED ===")
    return True


if __name__ == "__main__":
    ok = asyncio.run(smoke_test())
    sys.exit(0 if ok else 1)
