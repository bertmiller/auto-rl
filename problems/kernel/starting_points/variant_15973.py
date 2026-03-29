"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit(self, bundle):
        """Emit a VLIW bundle (dict of engine -> list of slots)."""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.emit({"load": [("const", addr, val)]})
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel: vectorized VLIW with aggressive scheduling.
        """
        # === Scratch allocation ===
        tmp1 = self.alloc_scratch("tmp1")

        # Header variables
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.emit({"load": [("const", tmp1, i)]})
            self.emit({"load": [("load", self.scratch[v], tmp1)]})

        # Constants
        zero_c = self.scratch_const(0)
        one_c = self.scratch_const(1)
        two_c = self.scratch_const(2)
        vlen_c = self.scratch_const(VLEN)

        # Hash constants (scalar)
        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_consts.append((op1, c1, op2, op3, c3))

        # Vector registers
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Vector hash constants
        v_hash_consts = []
        for si, (op1, c1, op2, op3, c3) in enumerate(hash_consts):
            vc1 = self.alloc_scratch(f"v_hc1_{si}", VLEN)
            vc3 = self.alloc_scratch(f"v_hc3_{si}", VLEN)
            v_hash_consts.append((op1, vc1, op2, op3, vc3))

        # Loop variables
        round_ctr = self.alloc_scratch("round_ctr")
        batch_ctr = self.alloc_scratch("batch_ctr")
        rounds_total = self.alloc_scratch("rounds_total")
        batch_total = self.alloc_scratch("batch_total")
        idx_base = self.alloc_scratch("idx_base")
        val_base = self.alloc_scratch("val_base")
        cond = self.alloc_scratch("cond")

        n_groups = batch_size // VLEN

        # Init loop bounds
        self.emit({"load": [
            ("const", rounds_total, rounds),
            ("const", batch_total, n_groups),
        ]})

        # Broadcast constants to vectors
        self.emit({"valu": [
            ("vbroadcast", v_zero, zero_c),
            ("vbroadcast", v_one, one_c),
            ("vbroadcast", v_two, two_c),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})
        # Broadcast hash constants (pack 2 broadcasts per cycle, limit 6 valu)
        for si in range(0, len(v_hash_consts), 3):
            slots = []
            for j in range(si, min(si+3, len(v_hash_consts))):
                slots.append(("vbroadcast", v_hash_consts[j][1], hash_consts[j][1]))
                slots.append(("vbroadcast", v_hash_consts[j][4], hash_consts[j][4]))
            self.emit({"valu": slots})

        # First pause
        self.emit({"flow": [("pause",)]})

        # === OUTER LOOP ===
        self.emit({"load": [("const", round_ctr, 0)]})
        round_loop_top = len(self.instrs)

        # === INNER LOOP ===
        self.emit({"load": [("const", batch_ctr, 0)]})
        batch_loop_top = len(self.instrs)

        # Cycle 1: addr calc + nothing else can run yet
        self.emit({"alu": [
            ("*", tmp1, batch_ctr, vlen_c),
        ]})
        # Cycle 2: base addr computation
        self.emit({"alu": [
            ("+", idx_base, self.scratch["inp_indices_p"], tmp1),
            ("+", val_base, self.scratch["inp_values_p"], tmp1),
        ]})
        # Cycle 3: vload idx + val (2 loads)
        self.emit({"load": [
            ("vload", v_idx, idx_base),
            ("vload", v_val, val_base),
        ]})
        # Cycle 4: broadcast forest_values_p to v_addr
        self.emit({"valu": [
            ("vbroadcast", v_addr, self.scratch["forest_values_p"]),
        ]})
        # Cycle 5: compute gather addresses
        self.emit({"valu": [
            ("+", v_addr, v_addr, v_idx),
        ]})

        # Cycles 6-9: Gather loads (2 per cycle, 4 cycles for 8 lanes)
        for lane_pair in range(0, VLEN, 2):
            self.emit({"load": [
                ("load_offset", v_node_val, v_addr, lane_pair),
                ("load_offset", v_node_val, v_addr, lane_pair + 1),
            ]})

        # Cycle 10: XOR val ^= node_val
        self.emit({"valu": [
            ("^", v_val, v_val, v_node_val),
        ]})

        # Cycles 11-22: Hash (6 stages × 2 cycles each)
        # Pack 2 independent valu ops per cycle for first half, 1 for merge
        for si, (op1, vc1, op2, op3, vc3) in enumerate(v_hash_consts):
            self.emit({"valu": [
                (op1, v_tmp1, v_val, vc1),
                (op3, v_tmp2, v_val, vc3),
            ]})
            self.emit({"valu": [
                (op2, v_val, v_tmp1, v_tmp2),
            ]})

        # Cycle 23: val & 1 (use & instead of %, same for power of 2)
        # and == 0 packed together? No, == depends on &. Do & first.
        self.emit({"valu": [
            ("&", v_tmp1, v_val, v_one),
        ]})
        # Cycle 24: == 0
        self.emit({"valu": [
            ("==", v_tmp1, v_tmp1, v_zero),
        ]})
        # Cycle 25: vselect (1 or 2) + pack idx*2 as shift in same cycle
        # flow: vselect; valu: idx << 1 (same as *2)
        self.emit({
            "flow": [("vselect", v_tmp3, v_tmp1, v_one, v_two)],
            "valu": [("<<", v_idx, v_idx, v_one)],
        })
        # Cycle 26: idx += offset
        self.emit({"valu": [
            ("+", v_idx, v_idx, v_tmp3),
        ]})
        # Cycle 27: idx < n_nodes
        self.emit({"valu": [
            ("<", v_tmp1, v_idx, v_n_nodes),
        ]})
        # Cycle 28: vselect wrap + store in same cycle? No, vselect writes v_idx, store reads it.
        # vselect first
        self.emit({"flow": [
            ("vselect", v_idx, v_tmp1, v_idx, v_zero),
        ]})
        # Cycle 29: store + inner loop increment (ALU can overlap with store)
        self.emit({
            "store": [
                ("vstore", idx_base, v_idx),
                ("vstore", val_base, v_val),
            ],
            "alu": [
                ("+", batch_ctr, batch_ctr, one_c),
            ],
        })
        # Cycle 30: compare + jump? No, cond_jump needs cond, so need compare first
        self.emit({"alu": [
            ("<", cond, batch_ctr, batch_total),
        ]})
        # Cycle 31: inner loop jump
        self.emit({"flow": [
            ("cond_jump", cond, batch_loop_top),
        ]})

        # Outer loop control
        # Can pack: round_ctr++, compare, jump over 2 cycles
        self.emit({"alu": [
            ("+", round_ctr, round_ctr, one_c),
            ("<", cond, round_ctr, rounds_total),
        ]})
        # Wait - round_ctr++ and < both reference round_ctr. In VLIW, reads happen before writes.
        # So < reads old round_ctr, not the incremented one. Need to separate.
        # Actually looking at Machine.step: scratch_write is a dict, writes happen AFTER all reads.
        # So ("+", round_ctr, round_ctr, one_c) and ("<", cond, round_ctr, rounds_total) in same cycle:
        # both read old round_ctr. The + writes new value, the < compares old value. That's wrong!
        # Need to fix: separate them.

        # Actually wait - let's think about this more carefully. After increment,
        # we need round_ctr < rounds_total. If we compare old value, we get
        # (old_round_ctr < rounds_total) which is true one extra time.
        # We need (old_round_ctr + 1 < rounds_total) i.e. (old_round_ctr < rounds_total - 1).
        # Hmm, simpler to just separate them.

        # Undo: remove the packed version and do sequential
        self.instrs.pop()  # remove the packed one

        self.emit({"alu": [
            ("+", round_ctr, round_ctr, one_c),
        ]})
        self.emit({"alu": [
            ("<", cond, round_ctr, rounds_total),
        ]})
        self.emit({"flow": [
            ("cond_jump", cond, round_loop_top),
        ]})

        # Final pause
        self.emit({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
