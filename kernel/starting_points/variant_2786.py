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
        """Emit a VLIW bundle: dict mapping engine -> list of slots"""
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
        Round-fused 8-chunk software-pipelined kernel with multiply_add.
        Overlaps round N+1 gathers with round N VALU tail.
        """
        NC = 16
        tmp1 = self.alloc_scratch("tmp1")

        # Init vars from memory header
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_c = self.scratch_const(0)
        one_c = self.scratch_const(1)
        two_c = self.scratch_const(2)
        vlen_c = self.scratch_const(VLEN)
        vlen_nc = self.scratch_const(VLEN * NC)

        # Hash stage constants
        hash_val1_c = []
        hash_val3_c = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_val1_c.append(self.scratch_const(val1))
            hash_val3_c.append(self.scratch_const(val3))

        # multiply_add constants for stages 0, 2, 4
        ma_mult_c = {}
        for hi in [0, 2, 4]:
            shift = HASH_STAGES[hi][4]
            ma_mult_c[hi] = self.scratch_const(1 + (1 << shift))

        # Offset constants for chunk pointers
        vlen_off = []
        for ci in range(NC):
            vlen_off.append(self.scratch_const(VLEN * ci))

        # Vector registers per chunk
        chunks = []
        for ci in range(NC):
            c = {}
            c['idx'] = self.alloc_scratch(f"c{ci}_idx", VLEN)
            c['val'] = self.alloc_scratch(f"c{ci}_val", VLEN)
            c['node'] = self.alloc_scratch(f"c{ci}_node", VLEN)
            c['tmp1'] = self.alloc_scratch(f"c{ci}_tmp1", VLEN)
            c['tmp2'] = self.alloc_scratch(f"c{ci}_tmp2", VLEN)
            c['cmp'] = self.alloc_scratch(f"c{ci}_cmp", VLEN)
            c['ga'] = self.alloc_scratch(f"c{ci}_ga", VLEN)
            chunks.append(c)

        # Shared vector constants
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_fvp = self.alloc_scratch("v_fvp", VLEN)

        # Hash constant vectors
        v_hc = {}
        v_sc = {}
        for hi in [1, 3, 5]:
            v_hc[hi] = self.alloc_scratch(f"v_hc_{hi}", VLEN)
            v_sc[hi] = self.alloc_scratch(f"v_sc_{hi}", VLEN)
        v_mm = {}
        v_ma = {}
        for hi in [0, 2, 4]:
            v_mm[hi] = self.alloc_scratch(f"v_mm_{hi}", VLEN)
            v_ma[hi] = self.alloc_scratch(f"v_ma_{hi}", VLEN)

        # Loop variables
        s_round_ctr = self.alloc_scratch("s_round_ctr")
        s_batch_ctr = self.alloc_scratch("s_batch_ctr")
        s_idx_base = self.alloc_scratch("s_idx_base")
        s_val_base = self.alloc_scratch("s_val_base")
        s_idx_addr = [self.alloc_scratch(f"s_idx_addr_{ci}") for ci in range(NC)]
        s_val_addr = [self.alloc_scratch(f"s_val_addr_{ci}") for ci in range(NC)]
        cond_tmp = self.alloc_scratch("cond_tmp")
        n_batches = batch_size // (VLEN * NC)

        # Pause
        self.add("flow", ("pause",))

        # === SETUP ===
        self.emit({"valu": [
            ("vbroadcast", v_two, two_c),
            ("vbroadcast", v_zero, zero_c),
            ("vbroadcast", v_one, one_c),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_fvp, self.scratch["forest_values_p"]),
        ]})
        self.emit({"valu": [
            ("vbroadcast", v_hc[1], hash_val1_c[1]),
            ("vbroadcast", v_sc[1], hash_val3_c[1]),
            ("vbroadcast", v_hc[3], hash_val1_c[3]),
            ("vbroadcast", v_sc[3], hash_val3_c[3]),
            ("vbroadcast", v_hc[5], hash_val1_c[5]),
            ("vbroadcast", v_sc[5], hash_val3_c[5]),
        ]})
        self.emit({"valu": [
            ("vbroadcast", v_mm[0], ma_mult_c[0]),
            ("vbroadcast", v_ma[0], hash_val1_c[0]),
            ("vbroadcast", v_mm[2], ma_mult_c[2]),
            ("vbroadcast", v_ma[2], hash_val1_c[2]),
            ("vbroadcast", v_mm[4], ma_mult_c[4]),
            ("vbroadcast", v_ma[4], hash_val1_c[4]),
        ]})

        self.emit({
            "load": [("const", s_batch_ctr, n_batches)],
            "alu": [
                ("+", s_idx_base, self.scratch["inp_indices_p"], zero_c),
                ("+", s_val_base, self.scratch["inp_values_p"], zero_c),
            ]
        })

        # === BATCH LOOP ===
        batch_loop_start = len(self.instrs)

        # Compute chunk pointers
        alu_ops = []
        for ci in range(NC):
            alu_ops.append(("+", s_idx_addr[ci], s_idx_base, vlen_off[ci]))
            alu_ops.append(("+", s_val_addr[ci], s_val_base, vlen_off[ci]))
        for alu_start in range(0, len(alu_ops), 12):
            self.emit({"alu": alu_ops[alu_start:alu_start+12]})

        # Load all chunks
        for ci in range(NC):
            self.emit({"load": [("vload", chunks[ci]['idx'], s_idx_addr[ci]),
                                ("vload", chunks[ci]['val'], s_val_addr[ci])]})

        self.emit({"load": [("const", s_round_ctr, rounds)]})

        # === ROUND LOOP ===
        # Use a proper scheduler that builds the entire schedule programmatically
        # and interleaves operations optimally.

        round_loop_start = len(self.instrs)

        # Per chunk, the operations needed each round (in dependency order):
        # 1. addr: ga = idx + v_fvp  (VALU, 1 slot, depends on idx)
        # 2. gather: load_offset(node, ga, 0..7) (LOAD, 8 slots = 4 cycles, depends on ga)
        # 3. xor: val = val ^ node  (VALU, 1 slot, depends on node)
        # 4. hash_s0: val = multiply_add(val, val, mm[0], ma[0]) (VALU, 1, dep on val from xor)
        # 5. hash_s1_par: tmp1 = op1(val,hc[1]), tmp2 = op3(val,sc[1]) (VALU, 2, dep on val from s0)
        # 6. hash_s1_comb: val = op2(tmp1,tmp2) (VALU, 1, dep on tmp1,tmp2)
        # 7. hash_s2: val = multiply_add(val, val, mm[2], ma[2]) (VALU, 1)
        # 8. hash_s3_par: tmp1 = op1(val,hc[3]), tmp2 = op3(val,sc[3]) (VALU, 2)
        # 9. hash_s3_comb: val = op2(tmp1,tmp2) (VALU, 1)
        # 10. hash_s4: val = multiply_add(val, val, mm[4], ma[4]) (VALU, 1)
        # 11. hash_s5_par: tmp1 = op1(val,hc[5]), tmp2 = op3(val,sc[5]) (VALU, 2)
        # 12. hash_s5_comb: val = op2(tmp1,tmp2) (VALU, 1)
        # 13. post_and: cmp = val & 1 (VALU, 1)
        # 14. post_dir: cmp = cmp + 1 (VALU, 1)
        # 15. post_idx: idx = multiply_add(idx, idx, v_two, cmp) (VALU, 1)
        # 16. post_lt: cmp = idx < n_nodes (VALU, 1)
        # 17. post_vsel: vselect(idx, cmp, idx, v_zero) (FLOW, 1)
        #
        # Total VALU per chunk: 1+1+1+2+1+1+2+1+1+2+1+1+1+1+1+1 = 20 VALU + 1 FLOW
        # Wait, let me recount: addr(1) + xor(1) + s0(1) + s1_par(2) + s1_comb(1) + s2(1) +
        #   s3_par(2) + s3_comb(1) + s4(1) + s5_par(2) + s5_comb(1) + and(1) + dir(1) + idx(1) + lt(1) = 19 VALU + 1 FLOW
        # Total LOAD per chunk: 8
        # Grand total: NC * 19 = 152 VALU, NC * 1 = 8 FLOW, NC * 8 = 64 LOAD

        # Define operations for each chunk
        # Each op: (engine, valu_slots, [chunk_idx], dep_phase)
        # We track per-chunk: ready_cycle (when the last dependency finishes)

        # Build a flat list of ops for all chunks
        ops_per_chunk = []
        for ci in range(NC):
            c = chunks[ci]
            ops = [
                # (label, engine, slots, n_load_slots, dep_labels)
                ('addr', 'valu', [("+", c['ga'], c['idx'], v_fvp)], 0, []),
                ('g01', 'load', [("load_offset", c['node'], c['ga'], 0), ("load_offset", c['node'], c['ga'], 1)], 2, ['addr']),
                ('g23', 'load', [("load_offset", c['node'], c['ga'], 2), ("load_offset", c['node'], c['ga'], 3)], 2, ['g01']),
                ('g45', 'load', [("load_offset", c['node'], c['ga'], 4), ("load_offset", c['node'], c['ga'], 5)], 2, ['g23']),
                ('g67', 'load', [("load_offset", c['node'], c['ga'], 6), ("load_offset", c['node'], c['ga'], 7)], 2, ['g45']),
                ('xor', 'valu', [("^", c['val'], c['val'], c['node'])], 0, ['g67']),
                ('s0', 'valu', [("multiply_add", c['val'], c['val'], v_mm[0], v_ma[0])], 0, ['xor']),
                ('s1p', 'valu', [(HASH_STAGES[1][0], c['tmp1'], c['val'], v_hc[1]),
                                 (HASH_STAGES[1][3], c['tmp2'], c['val'], v_sc[1])], 0, ['s0']),
                ('s1c', 'valu', [(HASH_STAGES[1][2], c['val'], c['tmp1'], c['tmp2'])], 0, ['s1p']),
                ('s2', 'valu', [("multiply_add", c['val'], c['val'], v_mm[2], v_ma[2])], 0, ['s1c']),
                ('s3p', 'valu', [(HASH_STAGES[3][0], c['tmp1'], c['val'], v_hc[3]),
                                 (HASH_STAGES[3][3], c['tmp2'], c['val'], v_sc[3])], 0, ['s2']),
                ('s3c', 'valu', [(HASH_STAGES[3][2], c['val'], c['tmp1'], c['tmp2'])], 0, ['s3p']),
                ('s4', 'valu', [("multiply_add", c['val'], c['val'], v_mm[4], v_ma[4])], 0, ['s3c']),
                ('s5p', 'valu', [(HASH_STAGES[5][0], c['tmp1'], c['val'], v_hc[5]),
                                 (HASH_STAGES[5][3], c['tmp2'], c['val'], v_sc[5])], 0, ['s4']),
                ('s5c', 'valu', [(HASH_STAGES[5][2], c['val'], c['tmp1'], c['tmp2'])], 0, ['s5p']),
                ('pa', 'valu', [("&", c['cmp'], c['val'], v_one)], 0, ['s5c']),
                ('pd', 'valu', [("+", c['cmp'], c['cmp'], v_one)], 0, ['pa']),
                ('pi', 'valu', [("multiply_add", c['idx'], c['idx'], v_two, c['cmp'])], 0, ['pd']),
                ('pl', 'valu', [("<", c['cmp'], c['idx'], v_n_nodes)], 0, ['pi']),
                ('pv', 'flow', [("vselect", c['idx'], c['cmp'], c['idx'], v_zero)], 0, ['pl']),
            ]
            ops_per_chunk.append(ops)

        # Flatten with chunk prefix
        all_ops = []
        for ci in range(NC):
            for label, engine, slots, n_load, deps in ops_per_chunk[ci]:
                full_label = f"c{ci}_{label}"
                full_deps = [f"c{ci}_{d}" for d in deps]
                all_ops.append((full_label, engine, slots, n_load, full_deps))

        # Schedule using list scheduling with resource constraints
        # Resources per cycle: LOAD=2, VALU=6, FLOW=1, ALU=12
        scheduled = {}  # label -> cycle
        remaining = {op[0]: op for op in all_ops}
        cycle = 0
        max_cycles = 100

        while remaining and cycle < max_cycles:
            load_used = 0
            valu_used = 0
            flow_used = 0
            load_slots = []
            valu_slots = []
            flow_slots = []

            # Find all ready ops (deps satisfied, scheduled in prev cycles)
            ready = []
            for label, (_, engine, slots, n_load, deps) in remaining.items():
                if all(d in scheduled and scheduled[d] < cycle for d in deps):
                    ready.append(label)

            # Priority: loads first (critical path), then VALU, then FLOW
            # Among loads, prioritize earlier chunks
            ready_loads = [r for r in ready if remaining[r][1] == 'load']
            ready_valu = [r for r in ready if remaining[r][1] == 'valu']
            ready_flow = [r for r in ready if remaining[r][1] == 'flow']

            scheduled_this_cycle = []

            for label in ready_loads:
                _, engine, slots, n_load, _ = remaining[label]
                if load_used + n_load <= 2:
                    load_used += n_load
                    load_slots.extend(slots)
                    scheduled_this_cycle.append(label)

            for label in ready_valu:
                _, engine, slots, n_load, _ = remaining[label]
                if valu_used + len(slots) <= 6:
                    valu_used += len(slots)
                    valu_slots.extend(slots)
                    scheduled_this_cycle.append(label)

            for label in ready_flow:
                _, engine, slots, n_load, _ = remaining[label]
                if flow_used + len(slots) <= 1:
                    flow_used += len(slots)
                    flow_slots.extend(slots)
                    scheduled_this_cycle.append(label)

            if not scheduled_this_cycle:
                # Nothing ready - should not happen if deps are correct
                cycle += 1
                continue

            for label in scheduled_this_cycle:
                scheduled[label] = cycle
                del remaining[label]

            # Emit the bundle
            bundle = {}
            if load_slots:
                bundle["load"] = load_slots
            if valu_slots:
                bundle["valu"] = valu_slots
            if flow_slots:
                bundle["flow"] = flow_slots
            if bundle:
                self.emit(bundle)
            cycle += 1

        # Round loop control
        self.emit({"alu": [("-", s_round_ctr, s_round_ctr, one_c)]})
        self.emit({"alu": [("<", cond_tmp, zero_c, s_round_ctr)]})
        self.emit({"flow": [("cond_jump", cond_tmp, round_loop_start)]})

        # Store all chunks
        for ci in range(NC):
            self.emit({"store": [("vstore", s_idx_addr[ci], chunks[ci]['idx']),
                                 ("vstore", s_val_addr[ci], chunks[ci]['val'])]})

        # Advance batch pointer
        self.emit({"alu": [
            ("+", s_idx_base, s_idx_base, vlen_nc),
            ("+", s_val_base, s_val_base, vlen_nc),
            ("-", s_batch_ctr, s_batch_ctr, one_c),
        ]})
        self.emit({"alu": [("<", cond_tmp, zero_c, s_batch_ctr)]})
        self.emit({"flow": [("cond_jump", cond_tmp, batch_loop_start)]})

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
