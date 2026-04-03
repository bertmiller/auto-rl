"""
Execute a real rollout: GPT-4o-mini optimizing the kernel for a few attempts.

Usage:
    cd problems/kernel
    python3 test_rollout.py
"""

import asyncio
import json
import re
import sys
import time

from openai import OpenAI

sys.path.insert(0, ".")
from kernel_opt_env.sandbox import (
    KernelOptSandboxPool,
    sandbox_read_file,
    sandbox_run_analysis,
    sandbox_run_command,
    sandbox_run_tests,
    sandbox_write_file,
    sandbox_install_variant,
)
from kernel_opt_env import sandbox as sandbox_mod
from kernel_opt_env.tools import READABLE_PATHS, WRITABLE_PATHS, _format_test_result, _format_analysis_result

SYSTEM_PROMPT = """\
You are a performance engineering agent. Your task is to optimize a kernel
running on a custom VLIW SIMD machine simulator, minimizing clock cycles.

The kernel performs a batched tree traversal with hashing. The baseline
implementation is a naive scalar version running in 147,734 cycles. Your
goal is to reduce this as much as possible.

The machine has these engines executing in parallel per cycle:
  - alu:   12 scalar ALU slots
  - valu:  6 vector ALU slots (VLEN=8 elements each)
  - load:  2 load slots (scalar or vector)
  - store: 2 store slots (scalar or vector)
  - flow:  1 flow control slot

Key: all slots in one instruction bundle execute simultaneously in one cycle.
Effects (writes) don't take effect until end of cycle.

Your tools:
  read_file(path)              — read perf_takehome.py, problem.py, or test output
  edit_file(path, content)     — rewrite perf_takehome.py (full file, not a diff)
  run_tests()                  — run submission tests, get correctness + cycle count
  run_analysis(script)         — run an inline Python script to analyze bottlenecks
  run_command(command)         — run a shell command (cp, ls, cat, diff, head, tail, wc, grep)

The optimization loop:
  1. Read perf_takehome.py and problem.py to understand the current state.
  2. Use run_analysis() to profile the kernel — inspect slot utilization,
     cycle distribution, instruction counts, or any other metric.
  3. Identify the bottleneck and form an optimization hypothesis.
  4. Edit perf_takehome.py to implement it.
  5. Run tests to verify correctness and measure cycles.
  6. If correct and faster: checkpoint with run_command('cp perf_takehome.py best.py').
     If incorrect or slower: restore with run_command('cp best.py perf_takehome.py')
     and try something else.
  7. Repeat.

Do NOT modify problem.py or anything in tests/. The submission tests use
a frozen copy of the simulator.

Optimization strategies to consider:
  - VLIW instruction packing (fill multiple slots per bundle)
  - SIMD vectorization (process 8 batch elements per vector op)
  - Loop construction with jump/cond_jump instead of full unrolling
  - Instruction scheduling to maximize parallelism
  - Precomputation and constant folding

You have 3 optimization attempts. Be systematic — understand what the
bottleneck is before changing code.
"""

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Use 'perf_takehome.py' for kernel code, 'problem.py' for ISA reference, 'test_output.log' for last test output.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Overwrite perf_takehome.py with new content. Full file, not a diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Complete new file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run submission tests. Returns correctness + cycle count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": "Run an inline Python analysis script in the sandbox. Can import from problem.py and perf_takehome.py. 30s timeout.",
            "parameters": {
                "type": "object",
                "properties": {"script": {"type": "string", "description": "Complete Python script"}},
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command. Allowed: cp, ls, cat, diff, head, tail, wc, grep.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

BASELINE_CYCLES = 147734
MAX_TURNS = 30
MAX_ATTEMPTS = 3


async def dispatch_tool(sid: str, name: str, args: dict) -> str:
    if name == "read_file":
        path = args.get("path", "")
        if path not in READABLE_PATHS:
            return f"Error: cannot read '{path}'. Allowed: {sorted(READABLE_PATHS)}"
        return await sandbox_read_file(sid, path)
    elif name == "edit_file":
        path = args.get("path", "")
        if path not in WRITABLE_PATHS:
            return f"Error: {path} is read-only. Writable: {sorted(WRITABLE_PATHS)}"
        await sandbox_write_file(sid, path, args["content"])
        return "OK"
    elif name == "run_tests":
        result = await sandbox_run_tests(sid)
        return _format_test_result(result)
    elif name == "run_analysis":
        from kernel_opt_env.sandbox import sandbox_run_analysis
        result = await sandbox_run_analysis(sid, args["script"], timeout_secs=30)
        return _format_analysis_result(result)
    elif name == "run_command":
        from kernel_opt_env.sandbox import sandbox_run_command
        result = await sandbox_run_command(sid, args["command"], timeout_secs=10)
        if not result["success"]:
            return f"Error: {result['output']}"
        return result["output"]
    return f"Unknown tool: {name}"


async def run_rollout():
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()
    sid = await pool.acquire()

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    MODEL = "Qwen/Qwen3-4B"

    print(f"{'='*60}")
    print(f"KERNEL OPTIMIZATION ROLLOUT ({MODEL})")
    print(f"Baseline: {BASELINE_CYCLES:,} cycles")
    print(f"Max attempts: {MAX_ATTEMPTS}")
    print(f"{'='*60}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Begin optimizing. Start by reading perf_takehome.py and problem.py to understand the architecture."},
    ]

    best_cycles = BASELINE_CYCLES
    num_attempts = 0
    turn = 0
    t0 = time.time()

    try:
        while turn < MAX_TURNS and num_attempts < MAX_ATTEMPTS:
            turn += 1
            print(f"--- Turn {turn} ---")

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOL_DEFS,
                    tool_choice="auto",
                    max_tokens=8192,
                    temperature=0.7,
                )
            except Exception as e:
                print(f"  API error: {e}")
                break

            choice = response.choices[0]
            msg = choice.message

            msg_dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)

            if msg.content:
                text = msg.content
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Model: {text}")

            if not msg.tool_calls:
                if choice.finish_reason == "stop":
                    print("  (model stopped)")
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                args_str = str(args)
                if len(args_str) > 150:
                    args_str = args_str[:150] + "..."
                print(f"  -> {name}({args_str})")

                result = await dispatch_tool(sid, name, args)

                result_str = result
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                print(f"  <- {result_str}")

                if name == "run_tests":
                    num_attempts += 1
                    if result.startswith("CORRECT"):
                        m = re.search(r"cycles=(\d+)", result)
                        if m:
                            cycles = int(m.group(1))
                            if cycles < best_cycles:
                                best_cycles = cycles
                                print(f"  *** NEW BEST: {cycles:,} cycles ({BASELINE_CYCLES/cycles:.1f}x) ***")

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            print()

    finally:
        await pool.release(sid)

    elapsed = time.time() - t0
    import math
    speedup = BASELINE_CYCLES / best_cycles if best_cycles < BASELINE_CYCLES else 1.0
    log2_speedup = math.log2(speedup) if speedup > 1 else 0.0

    print(f"{'='*60}")
    print(f"ROLLOUT COMPLETE ({elapsed:.1f}s)")
    print(f"  Best:    {best_cycles:,} cycles")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  log2:    {log2_speedup:.3f}")
    print(f"  Turns:   {turn}/{MAX_TURNS}")
    print(f"  Evals:   {num_attempts}/{MAX_ATTEMPTS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_rollout())
