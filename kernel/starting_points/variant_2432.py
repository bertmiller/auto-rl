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
        Round-fused multi-chunk kernel with multiply_add and programmatic scheduling.
        NC chunks processed through all rounds before moving to next batch.
        Uses multiply_add for hash stages 0, 2, 4.
        Programmatic scheduler packs VALU into LOAD bubbles.
        """
        NC = 32  # number of chunks processed in parallel
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

        # === Scalar constants ===
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vlen_const = self.scratch_const(VLEN)
        vlen_nc_const = self.scratch_const(VLEN * NC)

        # Hash stage constants
        hash_val1_consts = []
        hash_val3_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_val1_consts.append(self.scratch_const(val1))
            hash_val3_consts.append(self.scratch_const(val3))

        # multiply_add constants for stages 0, 2, 4
        ma_stages = [0, 2, 4]
        ma_mul_consts = {}
        for hi in ma_stages:
            _, val1, _, _, shift = HASH_STAGES[hi]
            mul_val = 1 + (1 << shift)
            ma_mul_consts[hi] = self.scratch_const(mul_val)

        # === Vector registers for each chunk (shared regs to save scratch) ===
        # node and tmp1 share the same address (node dead after XOR, before hash writes tmp1)
        # tmp2 and cmp share the same address (tmp2 dead after last hash combine, before post-hash writes cmp)
        chunks = []
        for ci in range(NC):
            c = {}
            c['idx'] = self.alloc_scratch(f"c{ci}_idx", VLEN)
            c['val'] = self.alloc_scratch(f"c{ci}_val", VLEN)
            c['node'] = self.alloc_scratch(f"c{ci}_nt", VLEN)  # shared: node, then tmp1
            c['tmp1'] = c['node']  # alias
            c['tmp2'] = self.alloc_scratch(f"c{ci}_tc", VLEN)  # shared: tmp2, then cmp
            c['cmp'] = c['tmp2']  # alias
            chunks.append(c)

        # Shared vector constants
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Hash constant vectors
        v_hash_val1 = []
        for hi in range(len(HASH_STAGES)):
            v_hash_val1.append(self.alloc_scratch(f"v_hv1_{hi}", VLEN))
        v_hash_val3 = []
        for hi in range(len(HASH_STAGES)):
            v_hash_val3.append(self.alloc_scratch(f"v_hv3_{hi}", VLEN))

        # multiply_add vector constants
        v_ma_mul = {}
        for hi in ma_stages:
            v_ma_mul[hi] = self.alloc_scratch(f"v_ma_mul_{hi}", VLEN)

        # Scalar loop variables
        s_round_ctr = self.alloc_scratch("s_round_ctr")
        s_batch_ctr = self.alloc_scratch("s_batch_ctr")
        s_idx_addr = [self.alloc_scratch(f"s_idx_addr_{ci}") for ci in range(NC)]
        s_val_addr = [self.alloc_scratch(f"s_val_addr_{ci}") for ci in range(NC)]
        cond_tmp = self.alloc_scratch("cond_tmp")
        n_batch_groups = batch_size // (VLEN * NC)

        # Gather address registers (4 sets, cycled across chunks)
        N_GA_SETS = min(4, NC)
        s_ga_base = []
        for si in range(N_GA_SETS):
            s_ga_base.append([self.alloc_scratch(f"s_ga{si}_{i}") for i in range(VLEN)])
        s_ga_sets = [s_ga_base[ci % N_GA_SETS] for ci in range(NC)]

        # Pause for reference kernel yield
        self.add("flow", ("pause",))

        # === Setup: broadcast constants ===
        self.emit({"valu": [
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ], "load": [("const", s_batch_ctr, n_batch_groups)]})

        self.emit({"valu": [("vbroadcast", v_hash_val1[hi], hash_val1_consts[hi]) for hi in range(6)]})
        self.emit({"valu": [("vbroadcast", v_hash_val3[hi], hash_val3_consts[hi]) for hi in range(6)]})
        self.emit({"valu": [("vbroadcast", v_ma_mul[hi], ma_mul_consts[hi]) for hi in ma_stages]})

        # === Set up initial pointers (incremental to save scratch) ===
        self.emit({"alu": [
            ("+", s_idx_addr[0], self.scratch["inp_indices_p"], zero_const),
            ("+", s_val_addr[0], self.scratch["inp_values_p"], zero_const),
        ]})
        # Compute addr[ci] = addr[ci-1] + VLEN incrementally
        # Due to dependency chain, we can only pipeline 2 per cycle
        for ci in range(1, NC):
            self.emit({"alu": [
                ("+", s_idx_addr[ci], s_idx_addr[ci-1], vlen_const),
                ("+", s_val_addr[ci], s_val_addr[ci-1], vlen_const),
            ]})

        # === Outer loop: batch groups ===
        batch_loop_start = len(self.instrs)

        # Load all idx and val vectors for NC chunks
        for ci in range(NC):
            self.emit({"load": [("vload", chunks[ci]['idx'], s_idx_addr[ci]),
                                ("vload", chunks[ci]['val'], s_val_addr[ci])]})

        # Initialize round counter
        self.emit({"load": [("const", s_round_ctr, rounds)]})

        # === Round loop (programmatically scheduled) ===
        round_loop_start = len(self.instrs)

        # Build operation dependency graph for one round of NC chunks
        # Then greedily schedule into VLIW bundles
        self._schedule_round(NC, chunks, s_ga_sets, v_hash_val1, v_hash_val3,
                            v_ma_mul, ma_stages, v_one, v_two, v_zero, v_n_nodes,
                            s_round_ctr, one_const, zero_const, cond_tmp)

        # Merge cond_jump into the last instruction if it has room for a FLOW slot
        last_instr = self.instrs[-1]
        if 'flow' not in last_instr or len(last_instr.get('flow', [])) < SLOT_LIMITS['flow']:
            if 'flow' not in last_instr:
                last_instr['flow'] = []
            last_instr['flow'].append(("cond_jump", cond_tmp, round_loop_start))
        else:
            self.emit({"flow": [("cond_jump", cond_tmp, round_loop_start)]})

        # Store results back
        for ci in range(0, NC, 1):
            self.emit({"store": [("vstore", s_idx_addr[ci], chunks[ci]['idx']),
                                 ("vstore", s_val_addr[ci], chunks[ci]['val'])]})

        # Batch loop control
        alu_ops = [("-", s_batch_ctr, s_batch_ctr, one_const)]
        for ci in range(NC):
            alu_ops.append(("+", s_idx_addr[ci], s_idx_addr[ci], vlen_nc_const))
            alu_ops.append(("+", s_val_addr[ci], s_val_addr[ci], vlen_nc_const))
        while alu_ops:
            self.emit({"alu": alu_ops[:12]})
            alu_ops = alu_ops[12:]
        self.emit({"alu": [("<", cond_tmp, zero_const, s_batch_ctr)]})
        self.emit({"flow": [("cond_jump", cond_tmp, batch_loop_start)]})

        # Final pause
        self.emit({"flow": [("pause",)]})

    def _schedule_round(self, NC, chunks, s_ga_sets, v_hash_val1, v_hash_val3,
                        v_ma_mul, ma_stages, v_one, v_two, v_zero, v_n_nodes,
                        s_round_ctr, one_const, zero_const, cond_tmp):
        """
        Programmatically schedule one round for NC chunks.
        Builds a list of ops with dependencies, then greedily schedules.
        """
        forest_values_p = self.scratch["forest_values_p"]

        # Op representation: (engine, slot_tuple, op_id)
        ops = []
        op_id = 0
        deps = {}  # op_id -> set of prerequisite op_ids

        # Track which op_id last wrote each address (for RAW deps)
        last_writer = {}  # addr -> op_id
        # Track which op_ids read each address since the last write (for WAR deps)
        readers = defaultdict(set)  # addr -> set of op_ids

        def add_op(engine, slot, reads, writes):
            nonlocal op_id
            oid = op_id
            op_id += 1
            my_deps = set()
            # RAW: depend on last writer of each read address
            for r in reads:
                if r in last_writer:
                    my_deps.add(last_writer[r])
            # WAW: depend on last writer of each write address
            for w in writes:
                if w in last_writer:
                    my_deps.add(last_writer[w])
            # WAR: depend on all readers of each write address (since last write)
            for w in writes:
                for reader_id in readers.get(w, set()):
                    my_deps.add(reader_id)
            my_deps.discard(oid)  # no self-dep
            deps[oid] = my_deps
            ops.append((engine, slot, oid))
            # Update tracking
            for w in writes:
                last_writer[w] = oid
                readers[w] = set()  # clear readers since we have a new writer
            for r in reads:
                readers[r].add(oid)
            return oid

        for ci in range(NC):
            ga = s_ga_sets[ci]
            c = chunks[ci]

            # 1. Gather address computation (8 ALU ops)
            idx_reads = set(range(c['idx'], c['idx'] + VLEN))
            ga_writes = set(ga)
            ga_deps = []
            for lane in range(VLEN):
                r = {forest_values_p, c['idx'] + lane}
                w = {ga[lane]}
                ga_deps.append(add_op("alu", ("+", ga[lane], forest_values_p, c['idx'] + lane), r, w))

            # 2. Gather loads (8 LOAD ops, each depends on its ga)
            gather_deps = []
            for lane in range(VLEN):
                r = {ga[lane]}
                w = {c['node'] + lane}
                oid = add_op("load", ("load", c['node'] + lane, ga[lane]), r, w)
                gather_deps.append(oid)

            # 3. XOR: val ^= node (depends on all gathers completing)
            xor_reads = set(range(c['val'], c['val'] + VLEN)) | set(range(c['node'], c['node'] + VLEN))
            xor_writes = set(range(c['val'], c['val'] + VLEN))
            xor_id = add_op("valu", ("^", c['val'], c['val'], c['node']), xor_reads, xor_writes)

            # 4. Hash stages
            prev_val_id = xor_id
            for hi in range(6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                val_addrs = set(range(c['val'], c['val'] + VLEN))
                tmp1_addrs = set(range(c['tmp1'], c['tmp1'] + VLEN))
                tmp2_addrs = set(range(c['tmp2'], c['tmp2'] + VLEN))

                if hi in ma_stages:
                    # multiply_add: val = val * mul_const + val1_const
                    ma_reads = val_addrs | set(range(v_ma_mul[hi], v_ma_mul[hi] + VLEN)) | set(range(v_hash_val1[hi], v_hash_val1[hi] + VLEN))
                    prev_val_id = add_op("valu",
                        ("multiply_add", c['val'], c['val'], v_ma_mul[hi], v_hash_val1[hi]),
                        ma_reads, val_addrs)
                else:
                    # 3-op: tmp1 = op1(val, const1), tmp2 = op3(val, const3), val = op2(tmp1, tmp2)
                    hv1 = set(range(v_hash_val1[hi], v_hash_val1[hi] + VLEN))
                    hv3 = set(range(v_hash_val3[hi], v_hash_val3[hi] + VLEN))

                    add_op("valu", (op1, c['tmp1'], c['val'], v_hash_val1[hi]),
                           val_addrs | hv1, tmp1_addrs)
                    add_op("valu", (op3, c['tmp2'], c['val'], v_hash_val3[hi]),
                           val_addrs | hv3, tmp2_addrs)
                    prev_val_id = add_op("valu", (op2, c['val'], c['tmp1'], c['tmp2']),
                                         tmp1_addrs | tmp2_addrs, val_addrs)

            # 5. Post-hash: &1, <<1, ==0, vselect, +idx, <n, vselect_wrap
            cmp_addrs = set(range(c['cmp'], c['cmp'] + VLEN))
            idx_addrs = set(range(c['idx'], c['idx'] + VLEN))
            one_addrs = set(range(v_one, v_one + VLEN))
            two_addrs = set(range(v_two, v_two + VLEN))
            zero_addrs = set(range(v_zero, v_zero + VLEN))
            nn_addrs = set(range(v_n_nodes, v_n_nodes + VLEN))
            tmp1_addrs = set(range(c['tmp1'], c['tmp1'] + VLEN))

            # Post-hash optimized: 4-cycle critical path
            # Step 1 (parallel): cmp = val & 1, tmp1 = 2*idx + 1
            and_id = add_op("valu", ("&", c['cmp'], c['val'], v_one),
                           val_addrs | one_addrs, cmp_addrs)
            ma_idx = add_op("valu", ("multiply_add", c['tmp1'], c['idx'], v_two, v_one),
                           idx_addrs | two_addrs | one_addrs, tmp1_addrs)
            # Step 2: idx = tmp1 + cmp = 2*idx + 1 + (val&1)
            add_idx_id = add_op("valu", ("+", c['idx'], c['tmp1'], c['cmp']),
                               tmp1_addrs | cmp_addrs, idx_addrs)
            # Step 3: cmp = (idx < n_nodes)
            lt_id = add_op("valu", ("<", c['cmp'], c['idx'], v_n_nodes),
                          idx_addrs | nn_addrs, cmp_addrs)
            # Step 4: idx = idx * cmp (0 if out of bounds)
            vsw_id = add_op("valu", ("*", c['idx'], c['idx'], c['cmp']),
                           idx_addrs | cmp_addrs, idx_addrs)

        # Add round counter decrement (no deps on chunk ops, can schedule early)
        dec_id = add_op("alu", ("-", s_round_ctr, s_round_ctr, one_const),
                       {s_round_ctr, one_const}, {s_round_ctr})
        # Compare (depends on dec)
        cmp_round_id = add_op("alu", ("<", cond_tmp, zero_const, s_round_ctr),
                              {zero_const, s_round_ctr}, {cond_tmp})

        # Schedule all ops using critical-path list scheduling
        n_ops = len(ops)

        # Build successor list
        successors = defaultdict(list)
        for i in range(n_ops):
            for d in deps[i]:
                successors[d].append(i)

        # Compute critical path length (depth from end) for each op
        # depth[i] = 1 + max(depth[succ] for succ in successors[i])
        depth = [0] * n_ops
        # Topological order (reverse)
        visited = [False] * n_ops
        topo_order = []
        def topo_dfs(node):
            stack = [(node, False)]
            while stack:
                n, processed = stack.pop()
                if processed:
                    topo_order.append(n)
                    continue
                if visited[n]:
                    continue
                visited[n] = True
                stack.append((n, True))
                for s in successors[n]:
                    if not visited[s]:
                        stack.append((s, False))
        for i in range(n_ops):
            if not visited[i]:
                topo_dfs(i)
        # Process in reverse topo order to compute depth
        for node in topo_order:
            max_succ_depth = 0
            for s in successors[node]:
                if depth[s] + 1 > max_succ_depth:
                    max_succ_depth = depth[s] + 1
            depth[node] = max_succ_depth

        dep_count = [len(deps[i]) for i in range(n_ops)]
        ready = []
        for i in range(n_ops):
            if dep_count[i] == 0:
                ready.append(i)

        while ready:
            bundle = defaultdict(list)
            slot_counts = defaultdict(int)
            used_this_cycle = []

            # Sort by critical path depth (higher depth = schedule first)
            # Within same depth, prioritize LOAD > ALU > VALU > FLOW
            ready.sort(key=lambda oid: (
                -depth[oid],  # higher depth first
                0 if ops[oid][0] == "load" else (1 if ops[oid][0] == "alu" else (2 if ops[oid][0] == "valu" else 3)),
            ))

            new_ready = []
            for oid in ready:
                engine, slot, _ = ops[oid]
                limit = SLOT_LIMITS.get(engine, 0)
                if slot_counts[engine] < limit:
                    bundle[engine].append(slot)
                    slot_counts[engine] += 1
                    used_this_cycle.append(oid)
                else:
                    new_ready.append(oid)

            ready = new_ready

            emit_bundle = {}
            for eng, slots in bundle.items():
                if slots:
                    emit_bundle[eng] = slots
            if emit_bundle:
                self.emit(emit_bundle)

            for oid in used_this_cycle:
                for succ in successors[oid]:
                    dep_count[succ] -= 1
                    if dep_count[succ] == 0:
                        ready.append(succ)


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
