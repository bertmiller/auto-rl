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


# local-mutation-263246
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
        Quad-chunk SIMD kernel: processes 4 VLEN-chunks (32 items) per iteration.
        Hash stages staggered across chunks to maximize VALU utilization.
        Gathers overlap with hash computation from other chunks.
        """
        NC = 4  # number of chunks
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
        vlen4_const = self.scratch_const(VLEN * NC)

        # Hash stage scalar constants
        hash_val1_consts = []
        hash_val3_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            hash_val1_consts.append(self.scratch_const(val1))
            hash_val3_consts.append(self.scratch_const(val3))

        # === Vector registers for each chunk ===
        chunks = []
        for ci in range(NC):
            c = {}
            c['idx'] = self.alloc_scratch(f"c{ci}_idx", VLEN)
            c['val'] = self.alloc_scratch(f"c{ci}_val", VLEN)
            c['node'] = self.alloc_scratch(f"c{ci}_node", VLEN)
            c['tmp1'] = self.alloc_scratch(f"c{ci}_tmp1", VLEN)
            c['tmp2'] = self.alloc_scratch(f"c{ci}_tmp2", VLEN)
            c['cmp'] = self.alloc_scratch(f"c{ci}_cmp", VLEN)
            chunks.append(c)

        # Shared vector constants
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Hash constant vectors
        v_hash_consts = []
        for hi in range(len(HASH_STAGES)):
            v_hash_consts.append(self.alloc_scratch(f"v_hc_{hi}", VLEN))
        v_shift_consts = []
        for hi in range(len(HASH_STAGES)):
            v_shift_consts.append(self.alloc_scratch(f"v_sc_{hi}", VLEN))

        # Scalar loop variables
        s_round_ctr = self.alloc_scratch("s_round_ctr")
        s_batch_ctr = self.alloc_scratch("s_batch_ctr")
        s_idx_addr = [self.alloc_scratch(f"s_idx_addr_{ci}") for ci in range(NC)]
        s_val_addr = [self.alloc_scratch(f"s_val_addr_{ci}") for ci in range(NC)]
        cond_tmp = self.alloc_scratch("cond_tmp")
        n_quad_chunks = batch_size // (VLEN * NC)

        # Gather address registers (shared - reused across chunks)
        s_ga = [self.alloc_scratch(f"s_ga{i}") for i in range(VLEN)]

        # Pause for reference kernel yield
        self.add("flow", ("pause",))

        # === Setup ===
        self.emit({"valu": [
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ], "load": [("const", s_round_ctr, rounds)]})

        self.emit({"valu": [("vbroadcast", v_hash_consts[hi], hash_val1_consts[hi]) for hi in range(6)]})
        self.emit({"valu": [("vbroadcast", v_shift_consts[hi], hash_val3_consts[hi]) for hi in range(6)]})

        # === Outer loop: rounds ===
        round_loop_start = len(self.instrs)

        # Initialize batch counter and pointers
        self.emit({
            "load": [("const", s_batch_ctr, n_quad_chunks)],
            "alu": [
                ("+", s_idx_addr[0], self.scratch["inp_indices_p"], zero_const),
                ("+", s_val_addr[0], self.scratch["inp_values_p"], zero_const),
            ]
        })
        # Set up chunk pointers: each chunk is VLEN apart
        self.emit({"alu": [
            ("+", s_idx_addr[1], s_idx_addr[0], vlen_const),
            ("+", s_val_addr[1], s_val_addr[0], vlen_const),
        ]})
        self.emit({"alu": [
            ("+", s_idx_addr[2], s_idx_addr[1], vlen_const),
            ("+", s_val_addr[2], s_val_addr[1], vlen_const),
            ("+", s_idx_addr[3], s_idx_addr[1], vlen_const),  # will fix below
            ("+", s_val_addr[3], s_val_addr[1], vlen_const),
        ]})
        # Fix: addr[3] = addr[2] + VLEN (but addr[2] was just written, RAW!)
        # Need to reconsider: addr[2] = addr[1] + VLEN, addr[3] = addr[2] + VLEN
        # But addr[2] is written in same cycle as we try to compute addr[3].
        # Fix: addr[3] = addr[1] + 2*VLEN

        # Actually let me rewrite this properly
        self.instrs.pop()  # remove the bad one

        vlen2_const = self.scratch_const(VLEN * 2)
        vlen3_const = self.scratch_const(VLEN * 3)

        self.emit({"alu": [
            ("+", s_idx_addr[1], s_idx_addr[0], vlen_const),
            ("+", s_val_addr[1], s_val_addr[0], vlen_const),
            ("+", s_idx_addr[2], s_idx_addr[0], vlen2_const),
            ("+", s_val_addr[2], s_val_addr[0], vlen2_const),
            ("+", s_idx_addr[3], s_idx_addr[0], vlen3_const),
            ("+", s_val_addr[3], s_val_addr[0], vlen3_const),
        ]})

        # === Inner loop: batch ===
        batch_loop_start = len(self.instrs)

        # --- Phase 1: Load all indices and values ---
        # C1: vload chunk 0 idx+val
        self.emit({"load": [("vload", chunks[0]['idx'], s_idx_addr[0]),
                            ("vload", chunks[0]['val'], s_val_addr[0])]})
        # C2: vload chunk 1 idx+val + chunk 0 gather addrs
        self.emit({
            "load": [("vload", chunks[1]['idx'], s_idx_addr[1]),
                     ("vload", chunks[1]['val'], s_val_addr[1])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[0]['idx'] + i) for i in range(VLEN)],
        })
        # C3: vload chunk 2 idx+val + chunk 1 gather addrs
        self.emit({
            "load": [("vload", chunks[2]['idx'], s_idx_addr[2]),
                     ("vload", chunks[2]['val'], s_val_addr[2])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[1]['idx'] + i) for i in range(VLEN)],
        })
        # Wait... chunk 1 idx was just loaded in C2. vload writes at end of cycle.
        # So chunk[1]['idx'] is NOT available until C3. But we're reading it in C3's ALU.
        # That's fine: vload in C2 writes at end of C2, so C3 reads the new values.

        # Actually no! The key insight: writes happen at end of cycle. So the vload
        # in C2 writes chunks[1]['idx'] at END of C2. In C3, when ALU reads
        # chunks[1]['idx'], it sees the values written at end of C2. So this is OK.
        # Wait, actually the machine processes C3 AFTER C2 completes (including writes).
        # Let me verify: step() applies scratch_write AFTER all slots execute. Then
        # the NEXT instruction (C3) sees the updated values. YES, this is correct.

        # C4: vload chunk 3 idx+val + chunk 0 gather 0,1
        self.emit({
            "load": [("vload", chunks[3]['idx'], s_idx_addr[3]),
                     ("vload", chunks[3]['val'], s_val_addr[3])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[2]['idx'] + i) for i in range(VLEN)],
        })
        # Same issue: chunk[2]['idx'] loaded in C3, available in C4. OK.
        # But wait, I'm also computing chunk 2's gather addrs, not loading chunk 0 gathers yet.
        # I need to load chunk 0's gathers first since those addresses are ready.

        # Let me restructure: compute gather addrs then do gathers
        # The issue is we can only load 2 things per cycle and we have 4 vloads + 32 gathers = 40 loads total

        # Restructured approach:
        # C1: vload c0_idx, c0_val (2 LOAD)
        # C2: vload c1_idx, c1_val (2 LOAD) + c0 gather addrs (8 ALU)
        # C3: c0 gather 0,1 (2 LOAD) + c1 gather addrs (8 ALU)
        # C4: c0 gather 2,3 (2 LOAD) + c2 vload?? No, need LOAD for vloads too.

        # Problem: vloads and gathers both use LOAD engine (max 2/cycle).
        # 4 vloads of idx + 4 vloads of val = 8 vloads = 4 cycles just for vloads.
        # 4 chunks × 8 gathers = 32 scalar loads = 16 cycles for gathers.
        # Total LOAD cycles needed: 4 + 16 = 20 cycles minimum.

        # But we need to compute gather addrs before we can gather. And gather addrs
        # depend on idx, which depends on vloads. So there's a dependency chain.

        # Let me re-plan this more carefully.

        # Actually let me pop the bad instructions and restart
        while len(self.instrs) > batch_loop_start:
            self.instrs.pop()

        # --- Restructured Phase 1: Staggered loads ---
        # The key constraint: vload idx → compute gather addrs (ALU) → gather (LOAD)
        # Each chunk needs: 2 vloads, 8 ALU addr computations, 8 scalar loads
        # We stagger: while gathering chunk N, we vload and compute addrs for chunk N+1

        # C1: vload c0 idx+val
        self.emit({"load": [("vload", chunks[0]['idx'], s_idx_addr[0]),
                            ("vload", chunks[0]['val'], s_val_addr[0])]})

        # C2: vload c1 idx+val + c0 gather addrs
        self.emit({
            "load": [("vload", chunks[1]['idx'], s_idx_addr[1]),
                     ("vload", chunks[1]['val'], s_val_addr[1])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[0]['idx'] + i) for i in range(VLEN)],
        })

        # C3: c0 gather 0,1 + c1 gather addrs
        self.emit({
            "load": [("load", chunks[0]['node'] + 0, s_ga[0]),
                     ("load", chunks[0]['node'] + 1, s_ga[1])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[1]['idx'] + i) for i in range(VLEN)],
        })
        # NOTE: c1 idx was vloaded in C2 (writes at end of C2), so C3 reads updated values. OK.
        # But wait: s_ga is being overwritten here for c1's addrs. We need c0's s_ga[2..7] still!
        # BUG: We're reusing s_ga for c1's addresses but haven't finished c0's gathers!

        # Fix: We need separate gather addr registers, OR gather all of c0 before computing c1 addrs.
        # Let's use 2 sets of gather addr registers to pipeline.

        # Actually, let me reconsider. With only 8 gather addr registers, I need to:
        # 1. Compute all 8 c0 addrs
        # 2. Gather all 8 c0 values (4 cycles)
        # 3. Compute all 8 c1 addrs (can overlap with c0 gathers if using separate registers)
        # 4. etc.

        # With 2 sets of gather addr registers (s_ga and s_gb):
        # C2: c0 addrs → s_ga
        # C3: c0 gather from s_ga + c1 addrs → s_gb
        #   But c0 needs 4 gather cycles (C3-C6), during which s_ga must remain stable
        #   c1 addrs → s_gb is fine (different regs)
        # C4-C6: c0 gather from s_ga (remaining 6 loads)
        # C7: c1 gather from s_gb + c2 vload (but c2 not loaded yet!)

        # Problem: We need to vload c2 before we can compute c2's gather addrs.
        # vloads also use LOAD engine. So we can't vload c2 while gathering c0.

        # Let me just do all vloads first, then all gathers.
        # C1-C4: 4 chunks × 2 vloads = 8 vloads at 2/cycle = 4 cycles
        # C5: c0 gather addrs (ALU)... but we've already computed them in C2!
        # No, we haven't done any ALU yet if we do all vloads first.

        # OK, simplest clean approach:
        # C1: vload c0
        # C2: vload c1
        # C3: vload c2
        # C4: vload c3
        # C5: c0 gather addrs (8 ALU)
        # C6: c0 gather 0,1 (2 LOAD) + c1 gather addrs (8 ALU)
        # C7: c0 gather 2,3 (2 LOAD) + c2 gather addrs (8 ALU)
        # C8: c0 gather 4,5 (2 LOAD) + c3 gather addrs (8 ALU)
        # C9: c0 gather 6,7 (2 LOAD)
        # C10: c1 gather 0,1 (2 LOAD) + XOR c0 (VALU)
        # ... etc.

        # But this wastes the LOAD engine in C1-C4 for vloads when we could overlap.
        # Actually C1 uses 2 LOADs for vload, C2 uses 2 LOADs for vload, etc. That's efficient.
        # We just can't do gather addrs until the idx is loaded.

        # Let me restart with clean staggered approach using 2 gather addr sets.

        while len(self.instrs) > batch_loop_start:
            self.instrs.pop()

        # Use 2 sets of gather address registers for pipelining
        s_gb = [self.alloc_scratch(f"s_gb{i}") for i in range(VLEN)]

        # ===== INNER LOOP =====
        # Phase 1: Load all chunk data (idx, val, node) with staggered gather

        # C1: vload c0 idx,val
        self.emit({"load": [("vload", chunks[0]['idx'], s_idx_addr[0]),
                            ("vload", chunks[0]['val'], s_val_addr[0])]})

        # C2: vload c1 idx,val + c0 gather addrs → s_ga
        self.emit({
            "load": [("vload", chunks[1]['idx'], s_idx_addr[1]),
                     ("vload", chunks[1]['val'], s_val_addr[1])],
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[0]['idx'] + i) for i in range(VLEN)],
        })

        # C3: c0 gather 0,1 from s_ga + c1 gather addrs → s_gb
        self.emit({
            "load": [("load", chunks[0]['node'] + 0, s_ga[0]),
                     ("load", chunks[0]['node'] + 1, s_ga[1])],
            "alu": [("+", s_gb[i], self.scratch["forest_values_p"], chunks[1]['idx'] + i) for i in range(VLEN)],
        })

        # C4: c0 gather 2,3 from s_ga + vload c2 idx,val
        # Wait, can't do vloads and gathers simultaneously if both use LOAD (max 2).
        # 2 scalar loads + 2 vloads = 4 LOAD slots. Exceeds limit!
        # So: c0 gather 2,3 only (2 LOAD)
        self.emit({
            "load": [("load", chunks[0]['node'] + 2, s_ga[2]),
                     ("load", chunks[0]['node'] + 3, s_ga[3])],
        })

        # C5: c0 gather 4,5 from s_ga
        self.emit({
            "load": [("load", chunks[0]['node'] + 4, s_ga[4]),
                     ("load", chunks[0]['node'] + 5, s_ga[5])],
        })

        # C6: c0 gather 6,7 from s_ga
        self.emit({
            "load": [("load", chunks[0]['node'] + 6, s_ga[6]),
                     ("load", chunks[0]['node'] + 7, s_ga[7])],
        })

        # C7: c1 gather 0,1 from s_gb + XOR c0 (c0 node all available now)
        self.emit({
            "load": [("load", chunks[1]['node'] + 0, s_gb[0]),
                     ("load", chunks[1]['node'] + 1, s_gb[1])],
            "valu": [("^", chunks[0]['val'], chunks[0]['val'], chunks[0]['node'])],
        })

        # C8: c1 gather 2,3 from s_gb + c0 hash s0 parallel
        op1, _, _, op3, _ = HASH_STAGES[0]
        self.emit({
            "load": [("load", chunks[1]['node'] + 2, s_gb[2]),
                     ("load", chunks[1]['node'] + 3, s_gb[3])],
            "valu": [(op1, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[0]),
                     (op3, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[0])],
        })

        # C9: c1 gather 4,5 + c0 hash s0 combine + vload c2
        # Can't do vload + gathers in same cycle. Just gathers + VALU.
        _, _, op2, _, _ = HASH_STAGES[0]
        self.emit({
            "load": [("load", chunks[1]['node'] + 4, s_gb[4]),
                     ("load", chunks[1]['node'] + 5, s_gb[5])],
            "valu": [(op2, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2'])],
        })

        # C10: c1 gather 6,7 + c0 hash s1 parallel
        op1, _, _, op3, _ = HASH_STAGES[1]
        self.emit({
            "load": [("load", chunks[1]['node'] + 6, s_gb[6]),
                     ("load", chunks[1]['node'] + 7, s_gb[7])],
            "valu": [(op1, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[1]),
                     (op3, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[1])],
        })

        # C11: vload c2 idx,val + c0 hash s1 combine + XOR c1
        _, _, op2, _, _ = HASH_STAGES[1]
        self.emit({
            "load": [("vload", chunks[2]['idx'], s_idx_addr[2]),
                     ("vload", chunks[2]['val'], s_val_addr[2])],
            "valu": [(op2, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2']),
                     ("^", chunks[1]['val'], chunks[1]['val'], chunks[1]['node'])],
        })

        # Now: c0 has done hash s0-s1 (4/12), c1 is XOR'd ready for hash, c2 just loaded
        # Need: c2 gather addrs, c2 gathers, c0 hash s2-s5, c1 hash s0-s5, c2 hash, c3 everything

        # C12: c2 gather addrs → s_ga + c0 s2 par + c1 s0 par
        op1_0, _, _, op3_0, _ = HASH_STAGES[0]
        op1_2, _, _, op3_2, _ = HASH_STAGES[2]
        self.emit({
            "alu": [("+", s_ga[i], self.scratch["forest_values_p"], chunks[2]['idx'] + i) for i in range(VLEN)],
            "valu": [
                (op1_2, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[2]),
                (op3_2, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[2]),
                (op1_0, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[0]),
                (op3_0, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[0]),
            ],
        })

        # C13: c2 gather 0,1 + c0 s2 comb + c1 s0 comb
        _, _, op2_2, _, _ = HASH_STAGES[2]
        _, _, op2_0, _, _ = HASH_STAGES[0]
        self.emit({
            "load": [("load", chunks[2]['node'] + 0, s_ga[0]),
                     ("load", chunks[2]['node'] + 1, s_ga[1])],
            "valu": [
                (op2_2, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2']),
                (op2_0, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
            ],
        })

        # C14: c2 gather 2,3 + c0 s3 par + c1 s1 par
        op1_3, _, _, op3_3, _ = HASH_STAGES[3]
        op1_1, _, _, op3_1, _ = HASH_STAGES[1]
        self.emit({
            "load": [("load", chunks[2]['node'] + 2, s_ga[2]),
                     ("load", chunks[2]['node'] + 3, s_ga[3])],
            "valu": [
                (op1_3, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[3]),
                (op3_3, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[3]),
                (op1_1, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[1]),
                (op3_1, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[1]),
            ],
        })

        # C15: c2 gather 4,5 + c0 s3 comb + c1 s1 comb
        _, _, op2_3, _, _ = HASH_STAGES[3]
        _, _, op2_1, _, _ = HASH_STAGES[1]
        self.emit({
            "load": [("load", chunks[2]['node'] + 4, s_ga[4]),
                     ("load", chunks[2]['node'] + 5, s_ga[5])],
            "valu": [
                (op2_3, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2']),
                (op2_1, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
            ],
        })

        # C16: c2 gather 6,7 + c0 s4 par + c1 s2 par
        op1_4, _, _, op3_4, _ = HASH_STAGES[4]
        op1_2b, _, _, op3_2b, _ = HASH_STAGES[2]
        self.emit({
            "load": [("load", chunks[2]['node'] + 6, s_ga[6]),
                     ("load", chunks[2]['node'] + 7, s_ga[7])],
            "valu": [
                (op1_4, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[4]),
                (op3_4, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[4]),
                (op1_2b, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[2]),
                (op3_2b, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[2]),
            ],
        })

        # C17: vload c3 idx,val + c0 s4 comb + c1 s2 comb + XOR c2
        _, _, op2_4, _, _ = HASH_STAGES[4]
        _, _, op2_2c, _, _ = HASH_STAGES[2]
        self.emit({
            "load": [("vload", chunks[3]['idx'], s_idx_addr[3]),
                     ("vload", chunks[3]['val'], s_val_addr[3])],
            "valu": [
                (op2_4, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2']),
                (op2_2c, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
                ("^", chunks[2]['val'], chunks[2]['val'], chunks[2]['node']),
            ],
        })
        # c2 node is all available from C16.

        # C18: c3 gather addrs → s_gb + c0 s5 par + c1 s3 par + c2 s0 par
        op1_5, _, _, op3_5, _ = HASH_STAGES[5]
        op1_3b, _, _, op3_3b, _ = HASH_STAGES[3]
        op1_0b, _, _, op3_0b, _ = HASH_STAGES[0]
        self.emit({
            "alu": [("+", s_gb[i], self.scratch["forest_values_p"], chunks[3]['idx'] + i) for i in range(VLEN)],
            "valu": [
                (op1_5, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[5]),
                (op3_5, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[5]),
                (op1_3b, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[3]),
                (op3_3b, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[3]),
            ],
        })
        # Wait, c2 s0 par needs 2 more VALU slots. That's 6 total. Check:
        # c0 s5 par: 2 VALU, c1 s3 par: 2 VALU, c2 s0 par: 2 VALU = 6 total. At limit!
        # But also have 8 ALU for c3 addrs. ALU and VALU are different engines. OK.
        # But wait, c3 idx was just vloaded in C17. The vload writes at end of C17.
        # C18 reads chunks[3]['idx'] which was written at end of C17. So C18 sees new values. OK.

        # Let me add c2 s0 par to C18
        self.instrs.pop()  # remove C18 without c2
        self.emit({
            "alu": [("+", s_gb[i], self.scratch["forest_values_p"], chunks[3]['idx'] + i) for i in range(VLEN)],
            "valu": [
                (op1_5, chunks[0]['tmp1'], chunks[0]['val'], v_hash_consts[5]),
                (op3_5, chunks[0]['tmp2'], chunks[0]['val'], v_shift_consts[5]),
                (op1_3b, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[3]),
                (op3_3b, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[3]),
                (op1_0b, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[0]),
                (op3_0b, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[0]),
            ],
        })
        # 6 VALU + 8 ALU. Both within limits!

        # C19: c3 gather 0,1 + c0 s5 comb + c1 s3 comb + c2 s0 comb
        _, _, op2_5, _, _ = HASH_STAGES[5]
        _, _, op2_3b, _, _ = HASH_STAGES[3]
        _, _, op2_0b, _, _ = HASH_STAGES[0]
        self.emit({
            "load": [("load", chunks[3]['node'] + 0, s_gb[0]),
                     ("load", chunks[3]['node'] + 1, s_gb[1])],
            "valu": [
                (op2_5, chunks[0]['val'], chunks[0]['tmp1'], chunks[0]['tmp2']),
                (op2_3b, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
                (op2_0b, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
            ],
        })
        # c0 is DONE with hash!

        # C20: c3 gather 2,3 + c1 s4 par + c2 s1 par + c0 post-hash (&1, <<1)
        op1_4b, _, _, op3_4b, _ = HASH_STAGES[4]
        op1_1b, _, _, op3_1b, _ = HASH_STAGES[1]
        self.emit({
            "load": [("load", chunks[3]['node'] + 2, s_gb[2]),
                     ("load", chunks[3]['node'] + 3, s_gb[3])],
            "valu": [
                (op1_4b, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[4]),
                (op3_4b, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[4]),
                (op1_1b, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[1]),
                (op3_1b, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[1]),
                ("&", chunks[0]['cmp'], chunks[0]['val'], v_one),
                ("<<", chunks[0]['idx'], chunks[0]['idx'], v_one),
            ],
        })
        # 6 VALU. At limit!

        # C21: c3 gather 4,5 + c1 s4 comb + c2 s1 comb + c0 ==0
        _, _, op2_4b, _, _ = HASH_STAGES[4]
        _, _, op2_1b, _, _ = HASH_STAGES[1]
        self.emit({
            "load": [("load", chunks[3]['node'] + 4, s_gb[4]),
                     ("load", chunks[3]['node'] + 5, s_gb[5])],
            "valu": [
                (op2_4b, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
                (op2_1b, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
                ("==", chunks[0]['cmp'], chunks[0]['cmp'], v_zero),
            ],
        })

        # C22: c3 gather 6,7 + c1 s5 par + c2 s2 par + c0 vselect
        op1_5b, _, _, op3_5b, _ = HASH_STAGES[5]
        op1_2c, _, _, op3_2c, _ = HASH_STAGES[2]
        self.emit({
            "load": [("load", chunks[3]['node'] + 6, s_gb[6]),
                     ("load", chunks[3]['node'] + 7, s_gb[7])],
            "valu": [
                (op1_5b, chunks[1]['tmp1'], chunks[1]['val'], v_hash_consts[5]),
                (op3_5b, chunks[1]['tmp2'], chunks[1]['val'], v_shift_consts[5]),
                (op1_2c, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[2]),
                (op3_2c, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[2]),
            ],
            "flow": [("vselect", chunks[0]['tmp1'], chunks[0]['cmp'], v_one, v_two)],
        })
        # 4 VALU + 1 FLOW. Fine.

        # C23: c1 s5 comb + c2 s2 comb + XOR c3 + c0 +idx
        _, _, op2_5b, _, _ = HASH_STAGES[5]
        _, _, op2_2d, _, _ = HASH_STAGES[2]
        self.emit({
            "valu": [
                (op2_5b, chunks[1]['val'], chunks[1]['tmp1'], chunks[1]['tmp2']),
                (op2_2d, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
                ("^", chunks[3]['val'], chunks[3]['val'], chunks[3]['node']),
                ("+", chunks[0]['idx'], chunks[0]['idx'], chunks[0]['tmp1']),
            ],
        })
        # c1 is DONE with hash! c3 is XOR'd.
        # 4 VALU.

        # C24: c2 s3 par + c3 s0 par + c0 <n + c1 &1,<<1
        op1_3c, _, _, op3_3c, _ = HASH_STAGES[3]
        op1_0c, _, _, op3_0c, _ = HASH_STAGES[0]
        self.emit({
            "valu": [
                (op1_3c, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[3]),
                (op3_3c, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[3]),
                (op1_0c, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[0]),
                (op3_0c, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[0]),
                ("<", chunks[0]['cmp'], chunks[0]['idx'], v_n_nodes),
                ("&", chunks[1]['cmp'], chunks[1]['val'], v_one),
            ],
        })
        # 6 VALU. At limit! But wait, c1 needs <<1 too. That's 7 VALU. EXCEEDS LIMIT!
        # Need to move c1 <<1 to next cycle.
        self.instrs.pop()
        self.emit({
            "valu": [
                (op1_3c, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[3]),
                (op3_3c, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[3]),
                (op1_0c, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[0]),
                (op3_0c, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[0]),
                ("<", chunks[0]['cmp'], chunks[0]['idx'], v_n_nodes),
                ("&", chunks[1]['cmp'], chunks[1]['val'], v_one),
            ],
        })

        # C25: c2 s3 comb + c3 s0 comb + c0 vselect_wrap + c1 <<1, ==0
        _, _, op2_3c, _, _ = HASH_STAGES[3]
        _, _, op2_0c, _, _ = HASH_STAGES[0]
        self.emit({
            "valu": [
                (op2_3c, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
                (op2_0c, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
                ("<<", chunks[1]['idx'], chunks[1]['idx'], v_one),
                ("==", chunks[1]['cmp'], chunks[1]['cmp'], v_zero),
            ],
            "flow": [("vselect", chunks[0]['idx'], chunks[0]['cmp'], chunks[0]['idx'], v_zero)],
        })
        # 4 VALU + 1 FLOW.
        # Wait: c1 &1 was computed in C24. c1 ==0 reads cmp which was written by &1 in C24.
        # C25 reads cmp written at end of C24. OK since C25 is next cycle.
        # But c1 ==0 needs the result of &1, and <<1 is independent.
        # Actually c1's ==0 depends on c1's &1 (both operate on cmp). &1 writes cmp at end of C24.
        # ==0 reads cmp in C25. That's fine - different cycles.
        # BUT: <<1 writes idx, and ==0 writes cmp. These are independent. Both in same cycle is OK.

        # C26: c0 store + c2 s4 par + c3 s1 par + c1 vselect
        op1_4c, _, _, op3_4c, _ = HASH_STAGES[4]
        op1_1c, _, _, op3_1c, _ = HASH_STAGES[1]
        self.emit({
            "store": [("vstore", s_idx_addr[0], chunks[0]['idx']),
                      ("vstore", s_val_addr[0], chunks[0]['val'])],
            "valu": [
                (op1_4c, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[4]),
                (op3_4c, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[4]),
                (op1_1c, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[1]),
                (op3_1c, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[1]),
            ],
            "flow": [("vselect", chunks[1]['tmp1'], chunks[1]['cmp'], v_one, v_two)],
        })
        # 2 STORE + 4 VALU + 1 FLOW. All within limits.
        # Wait: c1's cmp was written by ==0 in C25. C26 reads it for vselect. OK (next cycle).

        # C27: c2 s4 comb + c3 s1 comb + c1 +idx
        _, _, op2_4c, _, _ = HASH_STAGES[4]
        _, _, op2_1c, _, _ = HASH_STAGES[1]
        self.emit({
            "valu": [
                (op2_4c, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
                (op2_1c, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
                ("+", chunks[1]['idx'], chunks[1]['idx'], chunks[1]['tmp1']),
            ],
        })

        # C28: c2 s5 par + c3 s2 par + c1 <n
        op1_5c, _, _, op3_5c, _ = HASH_STAGES[5]
        op1_2d, _, _, op3_2d, _ = HASH_STAGES[2]
        self.emit({
            "valu": [
                (op1_5c, chunks[2]['tmp1'], chunks[2]['val'], v_hash_consts[5]),
                (op3_5c, chunks[2]['tmp2'], chunks[2]['val'], v_shift_consts[5]),
                (op1_2d, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[2]),
                (op3_2d, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[2]),
                ("<", chunks[1]['cmp'], chunks[1]['idx'], v_n_nodes),
            ],
        })
        # 5 VALU.

        # C29: c2 s5 comb + c3 s2 comb + c1 vselect_wrap
        _, _, op2_5c, _, _ = HASH_STAGES[5]
        _, _, op2_2e, _, _ = HASH_STAGES[2]
        self.emit({
            "valu": [
                (op2_5c, chunks[2]['val'], chunks[2]['tmp1'], chunks[2]['tmp2']),
                (op2_2e, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
            ],
            "flow": [("vselect", chunks[1]['idx'], chunks[1]['cmp'], chunks[1]['idx'], v_zero)],
        })
        # c2 is DONE with hash!

        # C30: c1 store + c3 s3 par + c2 &1,<<1
        op1_3d, _, _, op3_3d, _ = HASH_STAGES[3]
        self.emit({
            "store": [("vstore", s_idx_addr[1], chunks[1]['idx']),
                      ("vstore", s_val_addr[1], chunks[1]['val'])],
            "valu": [
                (op1_3d, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[3]),
                (op3_3d, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[3]),
                ("&", chunks[2]['cmp'], chunks[2]['val'], v_one),
                ("<<", chunks[2]['idx'], chunks[2]['idx'], v_one),
            ],
        })
        # 2 STORE + 4 VALU.

        # C31: c3 s3 comb + c2 ==0
        _, _, op2_3d, _, _ = HASH_STAGES[3]
        self.emit({
            "valu": [
                (op2_3d, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
                ("==", chunks[2]['cmp'], chunks[2]['cmp'], v_zero),
            ],
        })

        # C32: c3 s4 par + c2 vselect
        op1_4d, _, _, op3_4d, _ = HASH_STAGES[4]
        self.emit({
            "valu": [
                (op1_4d, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[4]),
                (op3_4d, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[4]),
            ],
            "flow": [("vselect", chunks[2]['tmp1'], chunks[2]['cmp'], v_one, v_two)],
        })

        # C33: c3 s4 comb + c2 +idx
        _, _, op2_4d, _, _ = HASH_STAGES[4]
        self.emit({
            "valu": [
                (op2_4d, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
                ("+", chunks[2]['idx'], chunks[2]['idx'], chunks[2]['tmp1']),
            ],
        })

        # C34: c3 s5 par + c2 <n
        op1_5d, _, _, op3_5d, _ = HASH_STAGES[5]
        self.emit({
            "valu": [
                (op1_5d, chunks[3]['tmp1'], chunks[3]['val'], v_hash_consts[5]),
                (op3_5d, chunks[3]['tmp2'], chunks[3]['val'], v_shift_consts[5]),
                ("<", chunks[2]['cmp'], chunks[2]['idx'], v_n_nodes),
            ],
        })

        # C35: c3 s5 comb + c2 vselect_wrap
        _, _, op2_5d, _, _ = HASH_STAGES[5]
        self.emit({
            "valu": [
                (op2_5d, chunks[3]['val'], chunks[3]['tmp1'], chunks[3]['tmp2']),
            ],
            "flow": [("vselect", chunks[2]['idx'], chunks[2]['cmp'], chunks[2]['idx'], v_zero)],
        })
        # c3 is DONE with hash!

        # C36: c2 store + c3 &1,<<1
        self.emit({
            "store": [("vstore", s_idx_addr[2], chunks[2]['idx']),
                      ("vstore", s_val_addr[2], chunks[2]['val'])],
            "valu": [
                ("&", chunks[3]['cmp'], chunks[3]['val'], v_one),
                ("<<", chunks[3]['idx'], chunks[3]['idx'], v_one),
            ],
        })

        # C37: c3 ==0
        self.emit({
            "valu": [("==", chunks[3]['cmp'], chunks[3]['cmp'], v_zero)],
        })

        # C38: c3 vselect
        self.emit({
            "flow": [("vselect", chunks[3]['tmp1'], chunks[3]['cmp'], v_one, v_two)],
        })

        # C39: c3 +idx
        self.emit({"valu": [("+", chunks[3]['idx'], chunks[3]['idx'], chunks[3]['tmp1'])]})

        # C40: c3 <n + ptr advance
        self.emit({
            "valu": [("<", chunks[3]['cmp'], chunks[3]['idx'], v_n_nodes)],
            "alu": [
                ("+", s_idx_addr[0], s_idx_addr[0], vlen4_const),
                ("+", s_val_addr[0], s_val_addr[0], vlen4_const),
                ("+", s_idx_addr[1], s_idx_addr[1], vlen4_const),
                ("+", s_val_addr[1], s_val_addr[1], vlen4_const),
                ("+", s_idx_addr[2], s_idx_addr[2], vlen4_const),
                ("+", s_val_addr[2], s_val_addr[2], vlen4_const),
                ("-", s_batch_ctr, s_batch_ctr, one_const),
            ],
        })

        # C41: c3 vselect_wrap + batch cond
        self.emit({
            "flow": [("vselect", chunks[3]['idx'], chunks[3]['cmp'], chunks[3]['idx'], v_zero)],
            "alu": [
                ("<", cond_tmp, zero_const, s_batch_ctr),
                ("+", s_idx_addr[3], s_idx_addr[3], vlen4_const),
                ("+", s_val_addr[3], s_val_addr[3], vlen4_const),
            ],
        })

        # C42: c3 store
        self.emit({
            "store": [("vstore", s_idx_addr[3], chunks[3]['idx']),
                      ("vstore", s_val_addr[3], chunks[3]['val'])],
        })
        # Wait: s_idx_addr[3] was advanced in C41, but we need to store to the OLD address!
        # BUG: Need to store BEFORE advancing the pointer.

        # Fix: Move ptr advance after stores. Let me restructure the tail.
        # Remove C40, C41, C42
        self.instrs.pop()  # C42
        self.instrs.pop()  # C41
        self.instrs.pop()  # C40

        # C40: c3 <n
        self.emit({
            "valu": [("<", chunks[3]['cmp'], chunks[3]['idx'], v_n_nodes)],
        })

        # C41: c3 vselect_wrap
        self.emit({
            "flow": [("vselect", chunks[3]['idx'], chunks[3]['cmp'], chunks[3]['idx'], v_zero)],
        })

        # C42: c3 store + batch dec
        self.emit({
            "store": [("vstore", s_idx_addr[3], chunks[3]['idx']),
                      ("vstore", s_val_addr[3], chunks[3]['val'])],
            "alu": [("-", s_batch_ctr, s_batch_ctr, one_const)],
        })

        # C43: ptr advance all + batch cond
        self.emit({
            "alu": [
                ("+", s_idx_addr[0], s_idx_addr[0], vlen4_const),
                ("+", s_val_addr[0], s_val_addr[0], vlen4_const),
                ("+", s_idx_addr[1], s_idx_addr[1], vlen4_const),
                ("+", s_val_addr[1], s_val_addr[1], vlen4_const),
                ("+", s_idx_addr[2], s_idx_addr[2], vlen4_const),
                ("+", s_val_addr[2], s_val_addr[2], vlen4_const),
                ("+", s_idx_addr[3], s_idx_addr[3], vlen4_const),
                ("+", s_val_addr[3], s_val_addr[3], vlen4_const),
                ("<", cond_tmp, zero_const, s_batch_ctr),
            ],
        })
        # 9 ALU ops. Within ALU limit of 12.
        # Wait: s_batch_ctr was decremented in C42 (writes at end of C42).
        # C43 reads s_batch_ctr for cond check. That's OK since C43 is next cycle.

        # C44: batch jump
        self.emit({"flow": [("cond_jump", cond_tmp, batch_loop_start)]})

        # End of round
        self.emit({"alu": [("-", s_round_ctr, s_round_ctr, one_const)]})
        self.emit({"alu": [("<", cond_tmp, zero_const, s_round_ctr)]})
        self.emit({"flow": [("cond_jump", cond_tmp, round_loop_start)]})

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
