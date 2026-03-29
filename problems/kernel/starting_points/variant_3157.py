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
        """Emit a VLIW bundle: dict of engine -> list of slots"""
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
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        4-group interleaved VLIW kernel with multiply_add hash optimization.
        - Batch-outer, round-inner loop structure
        - multiply_add for hash stages 0,2,4 (1 valu op instead of 3)
        - 4-group interleaving for maximal valu utilization
        """
        N_GROUPS = 4

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        vlen_const = self.scratch_const(VLEN)
        stride_const = self.scratch_const(N_GROUPS * VLEN)

        # Hash stage analysis:
        # Stage 0: ("+", 0x7ED55D16, "+", "<<", 12) → a = a*(1+2^12) + c = a*4097 + c → multiply_add
        # Stage 1: ("^", 0xC761C23C, "^", ">>", 19) → regular split/combine
        # Stage 2: ("+", 0x165667B1, "+", "<<", 5)  → a = a*(1+2^5) + c = a*33 + c → multiply_add
        # Stage 3: ("+", 0xD3A2646C, "^", "<<", 9)  → (a+c) ^ (a<<9) → regular (XOR combine)
        # Stage 4: ("+", 0xFD7046C5, "+", "<<", 3)  → a = a*(1+2^3) + c = a*9 + c → multiply_add
        # Stage 5: ("^", 0xB55A4F09, "^", ">>", 16) → regular split/combine

        # Scalar hash constants for non-madd stages
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            hash_consts.append((op1, c1, op2, op3, c3))

        # multiply_add multiplier constants
        madd_mul_0 = self.scratch_const(1 + (1 << 12))   # 4097
        madd_mul_2 = self.scratch_const(1 + (1 << 5))    # 33
        madd_mul_4 = self.scratch_const(1 + (1 << 3))    # 9

        class Group:
            pass

        groups = []
        for g in range(N_GROUPS):
            gr = Group()
            gr.idx = self.alloc_scratch(f"g{g}_idx", VLEN)
            gr.val = self.alloc_scratch(f"g{g}_val", VLEN)
            gr.nv = self.alloc_scratch(f"g{g}_nv", VLEN)
            gr.t1 = self.alloc_scratch(f"g{g}_t1", VLEN)
            gr.t2 = self.alloc_scratch(f"g{g}_t2", VLEN)
            gr.idp1 = self.alloc_scratch(f"g{g}_idp1", VLEN)
            gr.idx_ptr = self.alloc_scratch(f"g{g}_idxp")
            gr.val_ptr = self.alloc_scratch(f"g{g}_valp")
            groups.append(gr)

        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Vector constants for non-madd hash stages (stages 1, 3, 5)
        v_hash_consts = {}
        for stage in [1, 3, 5]:
            vc1 = self.alloc_scratch(None, VLEN)
            vc3 = self.alloc_scratch(None, VLEN)
            v_hash_consts[stage] = (vc1, vc3)

        # Vector constants for madd stages (stages 0, 2, 4)
        v_madd_mul = {}
        v_madd_add = {}
        for stage, mul_const in [(0, madd_mul_0), (2, madd_mul_2), (4, madd_mul_4)]:
            vm = self.alloc_scratch(None, VLEN)
            va = self.alloc_scratch(None, VLEN)
            v_madd_mul[stage] = vm
            v_madd_add[stage] = va

        s_ga = [self.alloc_scratch(f"ga_{i}") for i in range(VLEN)]
        s_ga2 = [self.alloc_scratch(f"ga2_{i}") for i in range(VLEN)]

        round_ctr = self.alloc_scratch("round_ctr")
        batch_ctr = self.alloc_scratch("batch_ctr")
        batch_limit = self.alloc_scratch("batch_limit")
        round_limit = self.alloc_scratch("round_limit")
        cond = self.alloc_scratch("cond")

        self.add("flow", ("pause",))

        # Init broadcasts
        self.emit({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})

        # Broadcast hash constants for non-madd stages
        for stage in [1, 3, 5]:
            vc1, vc3 = v_hash_consts[stage]
            self.emit({"valu": [
                ("vbroadcast", vc1, hash_consts[stage][1]),
                ("vbroadcast", vc3, hash_consts[stage][4]),
            ]})

        # Broadcast madd constants
        for stage, mul_const in [(0, madd_mul_0), (2, madd_mul_2), (4, madd_mul_4)]:
            vm = v_madd_mul[stage]
            va = v_madd_add[stage]
            self.emit({"valu": [
                ("vbroadcast", vm, mul_const),
                ("vbroadcast", va, hash_consts[stage][1]),  # addend is the stage constant
            ]})

        n_vector_groups = batch_size // VLEN
        assert n_vector_groups % N_GROUPS == 0
        self.emit({"load": [
            ("const", batch_limit, n_vector_groups // N_GROUPS),
            ("const", round_limit, rounds),
        ]})

        # Hash operation helpers
        def hash_madd(g_idx, stage):
            """multiply_add: val = val * mul + add (1 valu slot)"""
            gr = groups[g_idx]
            return [("multiply_add", gr.val, gr.val, v_madd_mul[stage], v_madd_add[stage])]

        def hash_split(g_idx, stage):
            """Regular hash split: 2 valu slots"""
            gr = groups[g_idx]
            vc1, vc3 = v_hash_consts[stage]
            return [
                (HASH_STAGES[stage][0], gr.t1, gr.val, vc1),
                (HASH_STAGES[stage][3], gr.t2, gr.val, vc3),
            ]

        def hash_combine(g_idx, stage):
            """Regular hash combine: 1 valu slot"""
            gr = groups[g_idx]
            return [(HASH_STAGES[stage][2], gr.val, gr.t1, gr.t2)]

        # === BATCH LOOP (outer) ===
        self.emit({
            "load": [("const", batch_ctr, 0)],
            "alu": [
                ("+", groups[0].idx_ptr, self.scratch["inp_indices_p"], zero_const),
                ("+", groups[0].val_ptr, self.scratch["inp_values_p"], zero_const),
            ]
        })
        alu_ops = []
        for g in range(1, N_GROUPS):
            gv = self.scratch_const(g * VLEN)
            alu_ops.append(("+", groups[g].idx_ptr, groups[0].idx_ptr, gv))
            alu_ops.append(("+", groups[g].val_ptr, groups[0].val_ptr, gv))
        self.emit({"alu": alu_ops})

        batch_loop_start = len(self.instrs)

        # Load all groups
        for g in range(N_GROUPS):
            self.emit({"load": [
                ("vload", groups[g].idx, groups[g].idx_ptr),
                ("vload", groups[g].val, groups[g].val_ptr),
            ]})

        # === ROUND LOOP (inner) - deep software pipelined ===
        # Pre-gather G0 (via s_ga) and G1 (via s_ga2) in prologue and tail.
        # Loop body starts with G0+G1 gathered, G2 addr in s_ga.
        # Tail overlaps NEXT G0+G1 gather + G2 addr with G2/G3 hash completion.
        # Uses two address arrays (s_ga, s_ga2) to avoid conflicts.

        self.emit({"load": [("const", round_ctr, 0)]})

        g0 = groups[0]
        g1 = groups[1]
        g2 = groups[2]
        g3 = groups[3]
        fvp = self.scratch["forest_values_p"]

        # === PROLOGUE: Gather G0 (s_ga) and G1 (s_ga2) for first round ===
        # P0: G0 addr → s_ga
        self.emit({"alu": [("+", s_ga[j], fvp, g0.idx + j) for j in range(VLEN)]})
        # P1: G0 loads[0,1] + G0 idp1_s1
        self.emit({
            "load": [("load", g0.nv + 0, s_ga[0]), ("load", g0.nv + 1, s_ga[1])],
            "valu": [("+", g0.idp1, g0.idx, g0.idx)],
        })
        # P2: G0 loads[2,3] + G0 idp1_s2
        self.emit({
            "load": [("load", g0.nv + 2, s_ga[2]), ("load", g0.nv + 3, s_ga[3])],
            "valu": [("+", g0.idp1, g0.idp1, v_one)],
        })
        # P3: G0 loads[4,5] + G1 addr → s_ga2
        self.emit({
            "load": [("load", g0.nv + 4, s_ga[4]), ("load", g0.nv + 5, s_ga[5])],
            "alu": [("+", s_ga2[j], fvp, g1.idx + j) for j in range(VLEN)],
        })
        # P4: G0 loads[6,7] + G1 loads[0,1] ... wait, 4 loads! Must split.
        # P4: G0 loads[6,7]
        self.emit({"load": [("load", g0.nv + 6, s_ga[6]), ("load", g0.nv + 7, s_ga[7])]})
        # P5: G1 loads[0,1] + G1 idp1_s1
        self.emit({
            "load": [("load", g1.nv + 0, s_ga2[0]), ("load", g1.nv + 1, s_ga2[1])],
            "valu": [("+", g1.idp1, g1.idx, g1.idx)],
        })
        # P6: G1 loads[2,3] + G1 idp1_s2
        self.emit({
            "load": [("load", g1.nv + 2, s_ga2[2]), ("load", g1.nv + 3, s_ga2[3])],
            "valu": [("+", g1.idp1, g1.idp1, v_one)],
        })
        # P7: G1 loads[4,5]
        self.emit({"load": [("load", g1.nv + 4, s_ga2[4]), ("load", g1.nv + 5, s_ga2[5])]})
        # P8: G1 loads[6,7] + G2 addr → s_ga
        self.emit({
            "load": [("load", g1.nv + 6, s_ga2[6]), ("load", g1.nv + 7, s_ga2[7])],
            "alu": [("+", s_ga[j], fvp, g2.idx + j) for j in range(VLEN)],
        })

        # === LOOP BODY START ===
        # State: G0 nv loaded + idp1 done, G1 nv loaded + idp1 done, s_ga has G2 addr
        round_loop_start = len(self.instrs)

        # C0: G0 XOR + G1 XOR + G2 loads[0,1] + G2 idp1_s1
        self.emit({
            "valu": [("^", g0.val, g0.val, g0.nv), ("^", g1.val, g1.val, g1.nv), ("+", g2.idp1, g2.idx, g2.idx)],
            "load": [("load", g2.nv + 0, s_ga[0]), ("load", g2.nv + 1, s_ga[1])],
        })
        # C1: G0 madd0 + G1 madd0 + G2 loads[2,3] + G2 idp1_s2
        self.emit({
            "valu": hash_madd(0, 0) + hash_madd(1, 0) + [("+", g2.idp1, g2.idp1, v_one)],
            "load": [("load", g2.nv + 2, s_ga[2]), ("load", g2.nv + 3, s_ga[3])],
        })
        # C2: G0 s1_split + G1 s1_split + G2 loads[4,5]
        self.emit({
            "valu": hash_split(0, 1) + hash_split(1, 1),
            "load": [("load", g2.nv + 4, s_ga[4]), ("load", g2.nv + 5, s_ga[5])],
        })
        # C3: G0 s1_comb + G1 s1_comb + G2 loads[6,7] + G3 addr → s_ga2
        self.emit({
            "valu": hash_combine(0, 1) + hash_combine(1, 1),
            "load": [("load", g2.nv + 6, s_ga[6]), ("load", g2.nv + 7, s_ga[7])],
            "alu": [("+", s_ga2[j], fvp, g3.idx + j) for j in range(VLEN)],
        })
        # C4: G0 madd2 + G1 madd2 + G2 XOR + G3 loads[0,1] + G3 idp1_s1
        self.emit({
            "valu": hash_madd(0, 2) + hash_madd(1, 2) + [("^", g2.val, g2.val, g2.nv), ("+", g3.idp1, g3.idx, g3.idx)],
            "load": [("load", g3.nv + 0, s_ga2[0]), ("load", g3.nv + 1, s_ga2[1])],
        })
        # C5: G0 s3_split + G1 s3_split + G2 madd0 + G3 loads[2,3] + G3 idp1_s2  [valu: 6]
        self.emit({
            "valu": hash_split(0, 3) + hash_split(1, 3) + hash_madd(2, 0) + [("+", g3.idp1, g3.idp1, v_one)],
            "load": [("load", g3.nv + 2, s_ga2[2]), ("load", g3.nv + 3, s_ga2[3])],
        })
        # C6: G0 s3_comb + G1 s3_comb + G2 s1_split + G3 loads[4,5]
        self.emit({
            "valu": hash_combine(0, 3) + hash_combine(1, 3) + hash_split(2, 1),
            "load": [("load", g3.nv + 4, s_ga2[4]), ("load", g3.nv + 5, s_ga2[5])],
        })
        # C7: G0 madd4 + G1 madd4 + G2 s1_comb + G3 loads[6,7]
        self.emit({
            "valu": hash_madd(0, 4) + hash_madd(1, 4) + hash_combine(2, 1),
            "load": [("load", g3.nv + 6, s_ga2[6]), ("load", g3.nv + 7, s_ga2[7])],
        })

        # --- All 16 loads done (C0-C7: 8 load cycles) ---

        # C8: G0 s5_split + G1 s5_split + G2 madd2 + G3 XOR  [valu: 6]
        self.emit({
            "valu": hash_split(0, 5) + hash_split(1, 5) + hash_madd(2, 2) + [("^", g3.val, g3.val, g3.nv)],
        })
        # C9: G0 s5_comb + G1 s5_comb + G2 s3_split + G3 madd0  [valu: 5]
        self.emit({
            "valu": hash_combine(0, 5) + hash_combine(1, 5) + hash_split(2, 3) + hash_madd(3, 0),
        })
        # C10: G0 &1 + G1 &1 + G2 s3_comb + G3 s1_split  [valu: 5]
        self.emit({
            "valu": [("&", g0.t1, g0.val, v_one), ("&", g1.t1, g1.val, v_one)] + hash_combine(2, 3) + hash_split(3, 1),
        })
        # C11: G0 idx_add + G1 idx_add + G2 madd4 + G3 s1_comb  [valu: 4]
        self.emit({
            "valu": [("+", g0.idx, g0.idp1, g0.t1), ("+", g1.idx, g1.idp1, g1.t1)] + hash_madd(2, 4) + hash_combine(3, 1),
        })
        # C12: G0 < + G1 < + G2 s5_split + G3 madd2  [valu: 5]
        self.emit({
            "valu": [("<", g0.t1, g0.idx, v_n_nodes), ("<", g1.t1, g1.idx, v_n_nodes)] + hash_split(2, 5) + hash_madd(3, 2),
        })
        # C13: G0 vselect + G2 s5_comb + G3 s3_split  [flow: 1, valu: 3]
        self.emit({
            "flow": [("vselect", g0.idx, g0.t1, g0.idx, v_zero)],
            "valu": hash_combine(2, 5) + hash_split(3, 3),
        })
        # C14: G1 vselect + G2 &1 + G3 s3_comb + NEXT G0 addr → s_ga  [flow: 1, valu: 2, alu: 8]
        self.emit({
            "flow": [("vselect", g1.idx, g1.t1, g1.idx, v_zero)],
            "valu": [("&", g2.t1, g2.val, v_one)] + hash_combine(3, 3),
            "alu": [("+", s_ga[j], fvp, g0.idx + j) for j in range(VLEN)],
        })
        # C15: G2 idx_add + G3 madd4 + G0 idp1_s1 + NEXT G0 loads[0,1] + NEXT G1 addr → s_ga2  [valu: 4, load: 2, alu: 8]
        self.emit({
            "valu": [("+", g2.idx, g2.idp1, g2.t1)] + hash_madd(3, 4) + [("+", g0.idp1, g0.idx, g0.idx)],
            "load": [("load", g0.nv + 0, s_ga[0]), ("load", g0.nv + 1, s_ga[1])],
            "alu": [("+", s_ga2[j], fvp, g1.idx + j) for j in range(VLEN)],
        })
        # C16: G2 < + G3 s5_split + G0 idp1_s2 + NEXT G0 loads[2,3] + round++  [valu: 4, load: 2, alu: 1]
        self.emit({
            "valu": [("<", g2.t1, g2.idx, v_n_nodes)] + hash_split(3, 5) + [("+", g0.idp1, g0.idp1, v_one)],
            "load": [("load", g0.nv + 2, s_ga[2]), ("load", g0.nv + 3, s_ga[3])],
            "alu": [("+", round_ctr, round_ctr, one_const)],
        })
        # C17: G2 vselect + G3 s5_comb + NEXT G0 loads[4,5] + cond  [flow: 1, valu: 1, load: 2, alu: 1]
        self.emit({
            "flow": [("vselect", g2.idx, g2.t1, g2.idx, v_zero)],
            "valu": hash_combine(3, 5),
            "load": [("load", g0.nv + 4, s_ga[4]), ("load", g0.nv + 5, s_ga[5])],
            "alu": [("<", cond, round_ctr, round_limit)],
        })
        # C18: G3 &1 + G1 idp1_s1 + NEXT G0 loads[6,7] + NEXT G2 addr → s_ga  [valu: 2, load: 2, alu: 8]
        # Note: loads read s_ga (G0 addr from C14), alu overwrites s_ga with G2 addr (reads-before-writes)
        self.emit({
            "valu": [("&", g3.t1, g3.val, v_one), ("+", g1.idp1, g1.idx, g1.idx)],
            "load": [("load", g0.nv + 6, s_ga[6]), ("load", g0.nv + 7, s_ga[7])],
            "alu": [("+", s_ga[j], fvp, g2.idx + j) for j in range(VLEN)],
        })
        # C19: G3 idx_add + G1 idp1_s2 + NEXT G1 loads[0,1]  [valu: 2, load: 2]
        self.emit({
            "valu": [("+", g3.idx, g3.idp1, g3.t1), ("+", g1.idp1, g1.idp1, v_one)],
            "load": [("load", g1.nv + 0, s_ga2[0]), ("load", g1.nv + 1, s_ga2[1])],
        })
        # C20: G3 < + NEXT G1 loads[2,3]  [valu: 1, load: 2]
        self.emit({
            "valu": [("<", g3.t1, g3.idx, v_n_nodes)],
            "load": [("load", g1.nv + 2, s_ga2[2]), ("load", g1.nv + 3, s_ga2[3])],
        })
        # C21: G3 vselect + NEXT G1 loads[4,5]  [flow: 1, load: 2]
        self.emit({
            "flow": [("vselect", g3.idx, g3.t1, g3.idx, v_zero)],
            "load": [("load", g1.nv + 4, s_ga2[4]), ("load", g1.nv + 5, s_ga2[5])],
        })
        # C22: NEXT G1 loads[6,7] + cond_jump  [load: 2, flow: 1]
        self.emit({
            "load": [("load", g1.nv + 6, s_ga2[6]), ("load", g1.nv + 7, s_ga2[7])],
            "flow": [("cond_jump", cond, round_loop_start)],
        })

        # Store results
        for g in range(N_GROUPS):
            self.emit({"store": [
                ("vstore", groups[g].idx_ptr, groups[g].idx),
                ("vstore", groups[g].val_ptr, groups[g].val),
            ]})

        # Batch loop control
        alu_ops = [("+", batch_ctr, batch_ctr, one_const)]
        for gr in groups:
            alu_ops.append(("+", gr.idx_ptr, gr.idx_ptr, stride_const))
            alu_ops.append(("+", gr.val_ptr, gr.val_ptr, stride_const))
        self.emit({"alu": alu_ops})
        self.emit({"alu": [("<", cond, batch_ctr, batch_limit)]})
        self.emit({"flow": [("cond_jump", cond, batch_loop_start)]})

        self.instrs.append({"flow": [("pause",)]})


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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
