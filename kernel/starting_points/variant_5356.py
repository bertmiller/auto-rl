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
        4-group interleaved VLIW kernel with multiply_add optimization.
        Uses multiply_add for hash stages 0,2,4 (where op1="+",op2="+",op3="<<").
        """
        V = VLEN

        # --- Scalar init ---
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        tmp1 = self.alloc_scratch("tmp1")
        for i, v in enumerate(init_vars):
            self.emit({"load": [("const", tmp1, i)]})
            self.emit({"load": [("load", self.scratch[v], tmp1)]})

        zero_s = self.scratch_const(0)
        one_s = self.scratch_const(1)
        two_s = self.scratch_const(2)

        vzero = self.alloc_scratch("vzero", V)
        vone = self.alloc_scratch("vone", V)
        vtwo = self.alloc_scratch("vtwo", V)
        self.emit({"valu": [("vbroadcast", vzero, zero_s)]})
        self.emit({"valu": [("vbroadcast", vone, one_s), ("vbroadcast", vtwo, two_s)]})

        # Hash vector constants
        # For MAC stages (0,2,4): store (vc_add, v_mul, is_mac=True)
        # For REG stages (1,3,5): store (vc1, vc3, is_mac=False)
        hash_info = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            is_mac = (op1 == "+" and op2 == "+" and op3 == "<<")
            if is_mac:
                mul_val = 1 + (1 << val3)
                self.scratch_const(val1)
                self.scratch_const(mul_val)
                vc_add = self.alloc_scratch(f"vh_add_{hi}", V)
                v_mul = self.alloc_scratch(f"vh_mul_{hi}", V)
                hash_info.append({"mac": True, "vc_add": vc_add, "v_mul": v_mul,
                                   "val1": val1, "mul_val": mul_val})
            else:
                self.scratch_const(val1)
                self.scratch_const(val3)
                vc1 = self.alloc_scratch(f"vh1_{hi}", V)
                vc3 = self.alloc_scratch(f"vh3_{hi}", V)
                hash_info.append({"mac": False, "vc1": vc1, "vc3": vc3,
                                   "val1": val1, "val3": val3,
                                   "op1": op1, "op2": op2, "op3": op3})

        for hi, info in enumerate(hash_info):
            if info["mac"]:
                self.emit({"valu": [
                    ("vbroadcast", info["vc_add"], self.const_map[info["val1"]]),
                    ("vbroadcast", info["v_mul"], self.const_map[info["mul_val"]]),
                ]})
            else:
                self.emit({"valu": [
                    ("vbroadcast", info["vc1"], self.const_map[info["val1"]]),
                    ("vbroadcast", info["vc3"], self.const_map[info["val3"]]),
                ]})

        vn_nodes = self.alloc_scratch("vn_nodes", V)
        self.emit({"valu": [("vbroadcast", vn_nodes, self.scratch["n_nodes"])]})
        self.emit({"flow": [("pause",)]})

        # --- Groups ---
        groups = []
        for g in "ABCD":
            idx = self.alloc_scratch(f"{g}_idx", V)
            val = self.alloc_scratch(f"{g}_val", V)
            nv = self.alloc_scratch(f"{g}_nv", V)
            t1 = self.alloc_scratch(f"{g}_t1", V)
            t2 = self.alloc_scratch(f"{g}_t2", V)
            ga = self.alloc_scratch(f"{g}_ga", V)
            groups.append({"idx": idx, "val": val, "nv": nv, "t1": t1, "t2": t2, "ga": ga})
        A, B, C, D = groups

        # Loop vars
        round_ctr = self.alloc_scratch("round_ctr")
        batch_ctr = self.alloc_scratch("batch_ctr")
        n_quads = batch_size // (V * 4)
        n_quads_const = self.scratch_const(n_quads)
        n_quads_m1_const = self.scratch_const(n_quads - 1)
        rounds_total = self.scratch_const(rounds)
        stride_const = self.scratch_const(V * 4)
        vlen_const = self.scratch_const(V)
        vlen2_const = self.scratch_const(V * 2)
        vlen3_const = self.scratch_const(V * 3)
        idx_ptr = self.alloc_scratch("idx_ptr")
        val_ptr = self.alloc_scratch("val_ptr")
        cmp_tmp = self.alloc_scratch("cmp_tmp")
        fvp = self.scratch["forest_values_p"]

        # Group pointers
        a_ip = self.alloc_scratch("a_ip")
        a_vp = self.alloc_scratch("a_vp")
        b_ip = self.alloc_scratch("b_ip")
        b_vp = self.alloc_scratch("b_vp")
        c_ip = self.alloc_scratch("c_ip")
        c_vp = self.alloc_scratch("c_vp")
        d_ip = self.alloc_scratch("d_ip")
        d_vp = self.alloc_scratch("d_vp")

        # Helper to emit hash stage for a group
        def hash_mac(g, hi):
            """Emit multiply_add hash stage: val = val * mul + add. Returns list of 1 valu op."""
            info = hash_info[hi]
            return [("multiply_add", g["val"], g["val"], info["v_mul"], info["vc_add"])]

        def hash_reg_ops(g, hi):
            """Emit regular hash stage independent ops. Returns list of 2 valu ops."""
            info = hash_info[hi]
            return [(info["op1"], g["t1"], g["val"], info["vc1"]),
                    (info["op3"], g["t2"], g["val"], info["vc3"])]

        def hash_reg_combine(g, hi):
            """Emit regular hash stage combine op. Returns list of 1 valu op."""
            info = hash_info[hi]
            return [(info["op2"], g["val"], g["t1"], g["t2"])]

        self.emit({"load": [("const", round_ctr, 0)]})

        # === OUTER LOOP ===
        round_loop_start = len(self.instrs)
        self.emit({
            "load": [("const", batch_ctr, 0)],
            "alu": [
                ("+", idx_ptr, self.scratch["inp_indices_p"], zero_s),
                ("+", val_ptr, self.scratch["inp_values_p"], zero_s),
            ],
        })

        # === INNER LOOP ===
        batch_loop_start = len(self.instrs)

        # Compute group pointers
        self.emit({"alu": [
            ("+", a_ip, idx_ptr, zero_s), ("+", a_vp, val_ptr, zero_s),
            ("+", b_ip, idx_ptr, vlen_const), ("+", b_vp, val_ptr, vlen_const),
            ("+", c_ip, idx_ptr, vlen2_const), ("+", c_vp, val_ptr, vlen2_const),
            ("+", d_ip, idx_ptr, vlen3_const), ("+", d_vp, val_ptr, vlen3_const),
        ]})

        # --- Phase 1: vloads + addr calc ---
        self.emit({"load": [("vload", A["idx"], a_ip), ("vload", A["val"], a_vp)]})
        self.emit({"load": [("vload", B["idx"], b_ip), ("vload", B["val"], b_vp)]})
        self.emit({
            "load": [("vload", C["idx"], c_ip), ("vload", C["val"], c_vp)],
            "alu": [("+", A["ga"] + k, fvp, A["idx"] + k) for k in range(V)],
        })
        self.emit({
            "load": [("vload", D["idx"], d_ip), ("vload", D["val"], d_vp)],
            "alu": [("+", B["ga"] + k, fvp, B["idx"] + k) for k in range(V)],
        })

        # --- Phase 2: A gathers + CD addr calc ---
        self.emit({
            "load": [("load", A["nv"] + 0, A["ga"] + 0), ("load", A["nv"] + 1, A["ga"] + 1)],
            "alu": [("+", C["ga"] + k, fvp, C["idx"] + k) for k in range(V)],
        })
        self.emit({
            "load": [("load", A["nv"] + 2, A["ga"] + 2), ("load", A["nv"] + 3, A["ga"] + 3)],
            "alu": [("+", D["ga"] + k, fvp, D["idx"] + k) for k in range(V)],
        })
        self.emit({"load": [("load", A["nv"] + 4, A["ga"] + 4), ("load", A["nv"] + 5, A["ga"] + 5)]})
        self.emit({"load": [("load", A["nv"] + 6, A["ga"] + 6), ("load", A["nv"] + 7, A["ga"] + 7)]})

        # --- Phase 3: A XOR + B gathers ---
        self.emit({
            "valu": [("^", A["val"], A["val"], A["nv"])],
            "load": [("load", B["nv"] + 0, B["ga"] + 0), ("load", B["nv"] + 1, B["ga"] + 1)],
        })

        # --- Phase 4: A hash stage 0 (MAC) + B gathers ---
        # A hash0: multiply_add (1 cycle) + B gather[2:3]
        self.emit({
            "valu": hash_mac(A, 0),
            "load": [("load", B["nv"] + 2, B["ga"] + 2), ("load", B["nv"] + 3, B["ga"] + 3)],
        })
        # A hash1 (REG): ops + B gather[4:5]
        self.emit({
            "valu": hash_reg_ops(A, 1),
            "load": [("load", B["nv"] + 4, B["ga"] + 4), ("load", B["nv"] + 5, B["ga"] + 5)],
        })
        # A hash1 combine + B gather[6:7]
        self.emit({
            "valu": hash_reg_combine(A, 1),
            "load": [("load", B["nv"] + 6, B["ga"] + 6), ("load", B["nv"] + 7, B["ga"] + 7)],
        })
        # A hash2 (MAC) + B XOR
        self.emit({"valu": hash_mac(A, 2) + [("^", B["val"], B["val"], B["nv"])]})

        # --- Phase 5: A(3-5) + B(0-2) interleaved hash + C gathers ---
        # A stages 3,4,5: REG, MAC, REG
        # B stages 0,1,2: MAC, REG, MAC
        # Schedule:
        # A3_ops + B0_mac + C gather[0:1]  (2+1=3 valu ≤6, + 2 loads)
        self.emit({
            "valu": hash_reg_ops(A, 3) + hash_mac(B, 0),
            "load": [("load", C["nv"] + 0, C["ga"] + 0), ("load", C["nv"] + 1, C["ga"] + 1)],
        })
        # A3_combine + B1_ops + C gather[2:3]
        self.emit({
            "valu": hash_reg_combine(A, 3) + hash_reg_ops(B, 1),
            "load": [("load", C["nv"] + 2, C["ga"] + 2), ("load", C["nv"] + 3, C["ga"] + 3)],
        })
        # A4_mac + B1_combine + C gather[4:5]
        self.emit({
            "valu": hash_mac(A, 4) + hash_reg_combine(B, 1),
            "load": [("load", C["nv"] + 4, C["ga"] + 4), ("load", C["nv"] + 5, C["ga"] + 5)],
        })
        # A5_ops + B2_mac + C gather[6:7]
        self.emit({
            "valu": hash_reg_ops(A, 5) + hash_mac(B, 2),
            "load": [("load", C["nv"] + 6, C["ga"] + 6), ("load", C["nv"] + 7, C["ga"] + 7)],
        })
        # A5_combine + C XOR
        self.emit({"valu": hash_reg_combine(A, 5) + [("^", C["val"], C["val"], C["nv"])]})
        # A HASH DONE!

        # --- Phase 6: B(3-5) + C(0-2) hash + D gathers ---
        # B3_ops + C0_mac + D gather[0:1]
        self.emit({
            "valu": hash_reg_ops(B, 3) + hash_mac(C, 0),
            "load": [("load", D["nv"] + 0, D["ga"] + 0), ("load", D["nv"] + 1, D["ga"] + 1)],
        })
        # B3_combine + C1_ops + D gather[2:3]
        self.emit({
            "valu": hash_reg_combine(B, 3) + hash_reg_ops(C, 1),
            "load": [("load", D["nv"] + 2, D["ga"] + 2), ("load", D["nv"] + 3, D["ga"] + 3)],
        })
        # B4_mac + C1_combine + D gather[4:5]
        self.emit({
            "valu": hash_mac(B, 4) + hash_reg_combine(C, 1),
            "load": [("load", D["nv"] + 4, D["ga"] + 4), ("load", D["nv"] + 5, D["ga"] + 5)],
        })
        # B5_ops + C2_mac + D gather[6:7]
        self.emit({
            "valu": hash_reg_ops(B, 5) + hash_mac(C, 2),
            "load": [("load", D["nv"] + 6, D["ga"] + 6), ("load", D["nv"] + 7, D["ga"] + 7)],
        })
        # B5_combine + D XOR + A idx start (&, *)
        # New idx: offset = (val & 1) + 1, idx = idx*2 + offset
        self.emit({"valu": hash_reg_combine(B, 5) + [
            ("^", D["val"], D["val"], D["nv"]),
            ("&", A["t1"], A["val"], vone), ("*", A["idx"], A["idx"], vtwo),
        ]})
        # B HASH DONE!

        # --- Phase 7: C(3-5) + D(0-2) hash + A&B idx ---
        # A: t1 has &1. Next: t1 += 1, then idx += t1
        # C3_ops + D0_mac + A t1+1
        self.emit({"valu": hash_reg_ops(C, 3) + hash_mac(D, 0) + [
            ("+", A["t1"], A["t1"], vone),
        ]})
        # C3_combine + D1_ops + A idx+=t1
        self.emit({"valu": hash_reg_combine(C, 3) + hash_reg_ops(D, 1) + [
            ("+", A["idx"], A["idx"], A["t1"]),
        ]})
        # C4_mac + D1_combine + A <
        self.emit({"valu": hash_mac(C, 4) + hash_reg_combine(D, 1) + [
            ("<", A["t1"], A["idx"], vn_nodes),
        ]})
        # C5_ops + D2_mac + A wrap + B idx start (&, *)
        self.emit({
            "valu": hash_reg_ops(C, 5) + hash_mac(D, 2) + [
                ("&", B["t1"], B["val"], vone), ("*", B["idx"], B["idx"], vtwo),
            ],
            "flow": [("vselect", A["idx"], A["t1"], A["idx"], vzero)],
        })
        # C5_combine + B t1+1
        self.emit({"valu": hash_reg_combine(C, 5) + [("+", B["t1"], B["t1"], vone)]})
        # C HASH DONE!

        # --- Phase 8: D(3-5) hash + B idx + A stores ---
        # D3_ops + B idx+=t1 + A stores
        self.emit({
            "valu": hash_reg_ops(D, 3) + [("+", B["idx"], B["idx"], B["t1"])],
            "store": [("vstore", a_ip, A["idx"]), ("vstore", a_vp, A["val"])],
        })
        # D3_combine + B <
        self.emit({"valu": hash_reg_combine(D, 3) + [("<", B["t1"], B["idx"], vn_nodes)]})
        # D4_mac + B wrap + C idx start (&, *)
        self.emit({
            "valu": hash_mac(D, 4) + [
                ("&", C["t1"], C["val"], vone), ("*", C["idx"], C["idx"], vtwo),
            ],
            "flow": [("vselect", B["idx"], B["t1"], B["idx"], vzero)],
        })
        # D5_ops + C t1+1 + B stores
        self.emit({
            "valu": hash_reg_ops(D, 5) + [("+", C["t1"], C["t1"], vone)],
            "store": [("vstore", b_ip, B["idx"]), ("vstore", b_vp, B["val"])],
        })
        # D5_combine + C idx+=t1
        self.emit({"valu": hash_reg_combine(D, 5) + [("+", C["idx"], C["idx"], C["t1"])]})
        # D HASH DONE!

        # --- Phase 9: C&D idx finish ---
        # C < + D idx start (&, *) + pointer advance
        self.emit({
            "valu": [("<", C["t1"], C["idx"], vn_nodes),
                     ("&", D["t1"], D["val"], vone), ("*", D["idx"], D["idx"], vtwo)],
            "alu": [
                ("+", idx_ptr, idx_ptr, stride_const),
                ("+", val_ptr, val_ptr, stride_const),
            ],
        })
        # C wrap + D t1+1
        self.emit({
            "valu": [("+", D["t1"], D["t1"], vone)],
            "flow": [("vselect", C["idx"], C["t1"], C["idx"], vzero)],
        })
        # D idx+=t1 + C stores
        self.emit({
            "valu": [("+", D["idx"], D["idx"], D["t1"])],
            "store": [("vstore", c_ip, C["idx"]), ("vstore", c_vp, C["val"])],
        })
        # D <
        self.emit({"valu": [("<", D["t1"], D["idx"], vn_nodes)]})
        # D wrap
        self.emit({"flow": [("vselect", D["idx"], D["t1"], D["idx"], vzero)]})
        # D stores + batch_ctr + compare
        self.emit({
            "store": [("vstore", d_ip, D["idx"]), ("vstore", d_vp, D["val"])],
            "alu": [("<", cmp_tmp, batch_ctr, n_quads_m1_const)],
            "flow": [("add_imm", batch_ctr, batch_ctr, 1)],
        })
        self.emit({"flow": [("cond_jump", cmp_tmp, batch_loop_start)]})

        # Outer loop: use same trick - compare old round_ctr < rounds-1
        rounds_m1_const = self.scratch_const(rounds - 1)
        self.emit({
            "alu": [("<", cmp_tmp, round_ctr, rounds_m1_const)],
            "flow": [("add_imm", round_ctr, round_ctr, 1)],
        })
        self.emit({"flow": [("cond_jump", cmp_tmp, round_loop_start)]})

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
