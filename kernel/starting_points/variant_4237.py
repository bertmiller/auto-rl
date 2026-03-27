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

        # === ROUND LOOP (inner) ===
        self.emit({"load": [("const", round_ctr, 0)]})
        round_loop_start = len(self.instrs)

        # The hash sequence per group with multiply_add optimization:
        # Stage 0: madd (1 op)
        # Stage 1: split(2) → combine(1) (2 cycles)
        # Stage 2: madd (1 op)
        # Stage 3: split(2) → combine(1) (2 cycles)
        # Stage 4: madd (1 op)
        # Stage 5: split(2) → combine(1) (2 cycles)
        # Total: 3 madd + 3×2 regular = 3 + 6 = 9 ops in 9 cycles
        # But madd only takes 1 cycle and 1 valu slot. So per group: 9 cycles

        # With 4 groups interleaved, I need to carefully schedule.
        # The dependency chain per group:
        #   madd0 → s1_split → s1_combine → madd2 → s3_split → s3_combine → madd4 → s5_split → s5_combine

        # Each step takes 1 cycle. With 4 groups staggered by 1 cycle:
        # G0: madd0  s1sp  s1co  madd2  s3sp  s3co  madd4  s5sp  s5co
        # G1:        madd0  s1sp  s1co   madd2  s3sp  s3co  madd4  s5sp  s5co
        # G2:               madd0  s1sp   s1co  madd2  s3sp  s3co  madd4  s5sp  s5co
        # G3:                      madd0  s1sp  s1co   madd2  s3sp  s3co  madd4  s5sp  s5co

        # VALU usage per cycle (steady state):
        # Cycle 3: G0.madd2(1) + G1.s1co(1) + G2.s1sp(2) + G3.madd0(1) = 5 ✓
        # Cycle 4: G0.s3sp(2) + G1.madd2(1) + G2.s1co(1) + G3.s1sp(2) = 6 ✓
        # Cycle 5: G0.s3co(1) + G1.s3sp(2) + G2.madd2(1) + G3.s1co(1) = 5 ✓
        # etc.

        # Total pipeline: 9 (G0's chain) + 3 (stagger) = 12 cycles for all 4 groups' hash
        # vs 15 cycles in the old schedule. Saves 3 cycles!

        # But we also need pre-hash (gather, XOR) and post-hash (&1, idx, <, vselect).
        # Let me build the full interleaved schedule.

        # === A: pre-hash ===
        g = groups[0]
        # A.ga_addr
        self.emit({"alu": [
            ("+", s_ga[j], self.scratch["forest_values_p"], g.idx + j)
            for j in range(VLEN)
        ]})
        # A.gather[0,1] + A.idx*2
        self.emit({
            "load": [("load", g.nv + 0, s_ga[0]), ("load", g.nv + 1, s_ga[1])],
            "valu": [("+", g.idp1, g.idx, g.idx)],
        })
        # A.gather[2,3] + A.idx*2+1
        self.emit({
            "load": [("load", g.nv + 2, s_ga[2]), ("load", g.nv + 3, s_ga[3])],
            "valu": [("+", g.idp1, g.idp1, v_one)],
        })
        # A.gather[4,5]
        self.emit({"load": [("load", g.nv + 4, s_ga[4]), ("load", g.nv + 5, s_ga[5])]})

        # A.gather[6,7] + B.ga_addr (s_ga overwrite OK: loads read old, ALU writes new)
        g1 = groups[1]
        self.emit({
            "load": [("load", g.nv + 6, s_ga[6]), ("load", g.nv + 7, s_ga[7])],
            "alu": [("+", s_ga[j], self.scratch["forest_values_p"], g1.idx + j) for j in range(VLEN)],
        })

        # A.XOR + B.gather[0,1] + B.idx*2
        self.emit({
            "valu": [("^", g.val, g.val, g.nv), ("+", g1.idp1, g1.idx, g1.idx)],
            "load": [("load", g1.nv + 0, s_ga[0]), ("load", g1.nv + 1, s_ga[1])],
        })

        # A.madd0 + B.gather[2,3] + B.idx*2+1
        self.emit({
            "valu": hash_madd(0, 0) + [("+", g1.idp1, g1.idp1, v_one)],
            "load": [("load", g1.nv + 2, s_ga[2]), ("load", g1.nv + 3, s_ga[3])],
        })

        # A.s1_split + B.gather[4,5]
        self.emit({
            "valu": hash_split(0, 1),
            "load": [("load", g1.nv + 4, s_ga[4]), ("load", g1.nv + 5, s_ga[5])],
        })

        # A.s1_comb + B.gather[6,7] + C.ga_addr
        g2 = groups[2]
        self.emit({
            "valu": hash_combine(0, 1),
            "load": [("load", g1.nv + 6, s_ga[6]), ("load", g1.nv + 7, s_ga[7])],
            "alu": [("+", s_ga[j], self.scratch["forest_values_p"], g2.idx + j) for j in range(VLEN)],
        })

        # A.madd2 + B.XOR + C.gather[0,1] + C.idx*2
        self.emit({
            "valu": hash_madd(0, 2) + [("^", g1.val, g1.val, g1.nv), ("+", g2.idp1, g2.idx, g2.idx)],
            "load": [("load", g2.nv + 0, s_ga[0]), ("load", g2.nv + 1, s_ga[1])],
        })

        # A.s3_split + B.madd0 + C.gather[2,3] + C.idx*2+1
        self.emit({
            "valu": hash_split(0, 3) + hash_madd(1, 0) + [("+", g2.idp1, g2.idp1, v_one)],
            "load": [("load", g2.nv + 2, s_ga[2]), ("load", g2.nv + 3, s_ga[3])],
        })  # 2+1+1 = 4 valu ✓

        # A.s3_comb + B.s1_split + C.gather[4,5]
        self.emit({
            "valu": hash_combine(0, 3) + hash_split(1, 1),
            "load": [("load", g2.nv + 4, s_ga[4]), ("load", g2.nv + 5, s_ga[5])],
        })  # 1+2 = 3 valu ✓

        # A.madd4 + B.s1_comb + C.gather[6,7] + D.ga_addr
        g3 = groups[3]
        self.emit({
            "valu": hash_madd(0, 4) + hash_combine(1, 1),
            "load": [("load", g2.nv + 6, s_ga[6]), ("load", g2.nv + 7, s_ga[7])],
            "alu": [("+", s_ga[j], self.scratch["forest_values_p"], g3.idx + j) for j in range(VLEN)],
        })  # 1+1 = 2 valu ✓

        # A.s5_split + B.madd2 + C.XOR + D.gather[0,1] + D.idx*2
        self.emit({
            "valu": hash_split(0, 5) + hash_madd(1, 2) + [("^", g2.val, g2.val, g2.nv), ("+", g3.idp1, g3.idx, g3.idx)],
            "load": [("load", g3.nv + 0, s_ga[0]), ("load", g3.nv + 1, s_ga[1])],
        })  # 2+1+1+1 = 5 valu ✓

        # A.s5_comb + B.s3_split + C.madd0 + D.gather[2,3] + D.idx*2+1
        self.emit({
            "valu": hash_combine(0, 5) + hash_split(1, 3) + hash_madd(2, 0) + [("+", g3.idp1, g3.idp1, v_one)],
            "load": [("load", g3.nv + 2, s_ga[2]), ("load", g3.nv + 3, s_ga[3])],
        })  # 1+2+1+1 = 5 valu ✓

        # A DONE. A.&1 + B.s3_comb + C.s1_split + D.gather[4,5]
        self.emit({
            "valu": [("&", groups[0].t1, groups[0].val, v_one)] + hash_combine(1, 3) + hash_split(2, 1),
            "load": [("load", g3.nv + 4, s_ga[4]), ("load", g3.nv + 5, s_ga[5])],
        })  # 1+1+2 = 4 valu ✓

        # A.idx_add + B.madd4 + C.s1_comb + D.gather[6,7]
        self.emit({
            "valu": [("+", groups[0].idx, groups[0].idp1, groups[0].t1)] + hash_madd(1, 4) + hash_combine(2, 1),
            "load": [("load", g3.nv + 6, s_ga[6]), ("load", g3.nv + 7, s_ga[7])],
        })  # 1+1+1 = 3 valu ✓

        # A.< + B.s5_split + C.madd2 + D.XOR
        self.emit({
            "valu": [("<", groups[0].t1, groups[0].idx, v_n_nodes)] + hash_split(1, 5) + hash_madd(2, 2) + [("^", g3.val, g3.val, g3.nv)],
        })  # 1+2+1+1 = 5 valu ✓

        # A.vselect + B.s5_comb + C.s3_split + D.madd0
        self.emit({
            "flow": [("vselect", groups[0].idx, groups[0].t1, groups[0].idx, v_zero)],
            "valu": hash_combine(1, 5) + hash_split(2, 3) + hash_madd(3, 0),
        })  # 1+2+1 = 4 valu + 1 flow ✓

        # B.&1 + C.s3_comb + D.s1_split
        self.emit({
            "valu": [("&", groups[1].t1, groups[1].val, v_one)] + hash_combine(2, 3) + hash_split(3, 1),
        })  # 1+1+2 = 4 valu ✓

        # B.idx_add + C.madd4 + D.s1_comb
        self.emit({
            "valu": [("+", groups[1].idx, groups[1].idp1, groups[1].t1)] + hash_madd(2, 4) + hash_combine(3, 1),
        })  # 1+1+1 = 3 valu ✓

        # B.< + C.s5_split + D.madd2
        self.emit({
            "valu": [("<", groups[1].t1, groups[1].idx, v_n_nodes)] + hash_split(2, 5) + hash_madd(3, 2),
        })  # 1+2+1 = 4 valu ✓

        # B.vselect + C.s5_comb + D.s3_split
        self.emit({
            "flow": [("vselect", groups[1].idx, groups[1].t1, groups[1].idx, v_zero)],
            "valu": hash_combine(2, 5) + hash_split(3, 3),
        })  # 1+2 = 3 valu + 1 flow ✓

        # C.&1 + D.s3_comb
        self.emit({
            "valu": [("&", groups[2].t1, groups[2].val, v_one)] + hash_combine(3, 3),
        })  # 1+1 = 2 valu ✓

        # C.idx_add + D.madd4
        self.emit({
            "valu": [("+", groups[2].idx, groups[2].idp1, groups[2].t1)] + hash_madd(3, 4),
        })  # 1+1 = 2 valu ✓

        # C.< + D.s5_split
        self.emit({
            "valu": [("<", groups[2].t1, groups[2].idx, v_n_nodes)] + hash_split(3, 5),
        })  # 1+2 = 3 valu ✓

        # C.vselect + D.s5_comb
        self.emit({
            "flow": [("vselect", groups[2].idx, groups[2].t1, groups[2].idx, v_zero)],
            "valu": hash_combine(3, 5),
        })  # 1 valu + 1 flow ✓

        # D post-hash + round control
        self.emit({
            "valu": [("&", groups[3].t1, groups[3].val, v_one)],
            "alu": [("+", round_ctr, round_ctr, one_const)],
        })
        self.emit({
            "valu": [("+", groups[3].idx, groups[3].idp1, groups[3].t1)],
            "alu": [("<", cond, round_ctr, round_limit)],
        })
        self.emit({"valu": [("<", groups[3].t1, groups[3].idx, v_n_nodes)]})
        self.emit({"flow": [("vselect", groups[3].idx, groups[3].t1, groups[3].idx, v_zero)]})
        self.emit({"flow": [("cond_jump", cond, round_loop_start)]})

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
