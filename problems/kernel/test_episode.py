"""
Single-episode test: run a Qwen3.5-7B model against the kernel-opt-env.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.5-7B \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --max-model-len 16384 --gpu-memory-utilization 0.90

    # Then run this script:
    python test_episode.py
"""

import asyncio
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, ".")
from kernel_opt_env.sandbox import (
    KernelOptSandboxPool,
    sandbox_read_file,
    sandbox_run_analysis,
    sandbox_run_command,
    sandbox_run_tests,
    sandbox_write_file,
)
from kernel_opt_env import sandbox as sandbox_mod

SYSTEM_PROMPT_TEMPLATE = """\
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

You have {max_attempts} optimization attempts. Be systematic — understand what the
bottleneck is before changing code.
"""

# Tool definitions for OpenAI-compatible API
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the current content of a file in the experiment sandbox. "
                "Use 'perf_takehome.py' to read the kernel code, "
                "'problem.py' to read the simulator and ISA reference, "
                "'test_output.log' to read the last test run output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the challenge root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Overwrite perf_takehome.py with new content. "
                "Only perf_takehome.py may be edited. problem.py and tests/ are read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Must be 'perf_takehome.py'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new file content. Full file, not a diff.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the submission tests and return correctness + cycle count. "
                "Uses the frozen simulator to prevent cheating."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Run an inline Python analysis script to inspect the kernel's behavior. "
                "The script executes in the sandbox with access to problem.py and "
                "perf_takehome.py imports. 30-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "A complete Python script as a string.",
                    }
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the sandbox directory. "
                "Allowed: cp, ls, cat, diff, head, tail, wc, grep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

READABLE_PATHS = {"perf_takehome.py", "problem.py", "test_output.log", "tests/submission_tests.py", "best.py"}

MAX_TURNS = 40
MAX_ATTEMPTS = 5  # keep it short for testing
BASELINE_CYCLES = 147734


async def dispatch_tool(sandbox_id: str, name: str, args: dict) -> str:
    """Execute a tool call and return the result string."""
    if name == "read_file":
        path = args.get("path", "")
        if path not in READABLE_PATHS:
            return f"Error: cannot read '{path}'. Allowed: {sorted(READABLE_PATHS)}"
        return await sandbox_read_file(sandbox_id, path)

    elif name == "edit_file":
        path = args.get("path", "")
        if path != "perf_takehome.py":
            return f"Error: {path} is read-only. Only perf_takehome.py may be edited."
        await sandbox_write_file(sandbox_id, path, args["content"])
        return "OK"

    elif name == "run_tests":
        result = await sandbox_run_tests(sandbox_id)
        if result["crashed"]:
            return f"CRASH | last 50 lines:\n{result['output']}"
        if not result["correct"]:
            return f"INCORRECT | cycles={result['cycles']} | output:\n{result['output'][-500:]}"
        return (
            f"CORRECT | cycles={result['cycles']} | "
            f"speedup={result['speedup']}x | "
            f"tests_passed={result['tests_passed']}/{result['total_tests']}"
        )

    elif name == "run_analysis":
        result = await sandbox_run_analysis(sandbox_id, args["script"], timeout_secs=30)
        if not result["success"]:
            return f"ERROR:\n{result['output']}"
        return result["output"]

    elif name == "run_command":
        result = await sandbox_run_command(sandbox_id, args["command"], timeout_secs=10)
        if not result["success"]:
            return f"ERROR: {result['output']}"
        return result["output"]

    else:
        return f"Unknown tool: {name}"


async def run_episode():
    # Initialize sandbox
    pool = KernelOptSandboxPool(size=1)
    sandbox_mod._pool = pool
    await pool.initialize()
    sandbox_id = await pool.acquire()

    print(f"{'='*60}")
    print(f"KERNEL OPTIMIZATION EPISODE")
    print(f"Baseline: {BASELINE_CYCLES} cycles")
    print(f"Max attempts: {MAX_ATTEMPTS}")
    print(f"{'='*60}\n")

    # Connect to vLLM server
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_attempts=MAX_ATTEMPTS)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Begin optimizing the kernel. Start by reading the code and understanding the architecture."},
    ]

    best_cycles = BASELINE_CYCLES
    num_attempts = 0
    turn = 0

    try:
        while turn < MAX_TURNS and num_attempts < MAX_ATTEMPTS:
            turn += 1
            print(f"--- Turn {turn} ---")

            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen3-4B",
                    messages=messages,
                    tools=TOOL_DEFS,
                    tool_choice="auto",
                    max_tokens=4096,
                    temperature=0.7,
                )
            except Exception as e:
                print(f"API error: {e}")
                break

            choice = response.choices[0]
            assistant_msg = choice.message

            # Build assistant message dict for history
            msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg_dict)

            # Print assistant's reasoning
            if assistant_msg.content:
                text = assistant_msg.content
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Model: {text}")

            # If no tool calls, model is done thinking or done entirely
            if not assistant_msg.tool_calls:
                if choice.finish_reason == "stop":
                    print("  (model stopped without tool call)")
                continue

            # Execute tool calls
            for tc in assistant_msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Truncate logged args for readability
                args_summary = str(args)
                if len(args_summary) > 200:
                    args_summary = args_summary[:200] + "..."
                print(f"  Tool: {name}({args_summary})")

                result = await dispatch_tool(sandbox_id, name, args)

                # Truncate logged result
                result_summary = result
                if len(result_summary) > 300:
                    result_summary = result_summary[:300] + "..."
                print(f"  Result: {result_summary}")

                # Track optimization attempts
                if name == "run_tests":
                    num_attempts += 1
                    if result.startswith("CORRECT"):
                        import re
                        m = re.search(r"cycles=(\d+)", result)
                        if m:
                            cycles = int(m.group(1))
                            if cycles < best_cycles:
                                best_cycles = cycles
                                print(f"  *** NEW BEST: {cycles} cycles ({BASELINE_CYCLES/cycles:.1f}x speedup) ***")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            print()

    finally:
        await pool.release(sandbox_id)
        await pool.shutdown()

    # Summary
    import math
    speedup = BASELINE_CYCLES / best_cycles if best_cycles < BASELINE_CYCLES else 1.0
    log2_speedup = math.log2(speedup) if speedup > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"EPISODE COMPLETE")
    print(f"  Best cycles: {best_cycles}")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  log2 reward: {log2_speedup:.3f}")
    print(f"  Attempts used: {num_attempts}/{MAX_ATTEMPTS}")
    print(f"  Turns used: {turn}/{MAX_TURNS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(run_episode())
