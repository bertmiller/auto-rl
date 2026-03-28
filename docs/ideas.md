# Optimization Challenge Ideas

RL environments where an agent iteratively edits code to optimize a scalar metric,
evaluated by a cheap deterministic oracle.

## Design criteria

Every environment should have:
1. **Cheap deterministic oracle** — CPU-only, seconds not minutes, no stochasticity
2. **Code as the action** — agent writes/edits a solution file
3. **Continuous improvement gradient** — scalar metric to push on, not binary pass/fail
4. **Wide difficulty range** — naive solution exists, but 100x+ improvement is possible
5. **Resistance to memorization** — large enough solution space that you can't just memorize

## Existing

- **Kernel optimization** (`kernel/`) — optimize a VLIW SIMD kernel, metric is clock cycles

## Ideas

### Circle packing
Pack N circles into a unit square, maximizing total area covered.
- **Metric:** area utilization (covered area / square area)
- **Oracle cost:** ~ms (geometric intersection checks)
- **Why interesting:** continuous optimization, geometric reasoning, known optimal
  solutions for small N to benchmark against, scales difficulty by increasing N
- **Action:** agent writes a function returning circle positions and radii

### Automated market maker (AMM)
Design a bonding curve and fee structure for a simulated AMM, optimizing against
a historical or synthetic trade sequence.
- **Metric:** LP PnL, impermanent loss, or a composite (PnL - slippage - IL)
- **Oracle cost:** ~ms (replay trade sequence through simulator)
- **Why interesting:** financial domain, multi-objective tradeoffs, real-world
  applicable, can vary difficulty by changing trade distribution
- **Action:** agent writes the bonding curve / fee / rebalancing logic

### Sorting networks
Design a comparator network that sorts N elements, minimizing depth or comparator count.
- **Metric:** network depth or total comparators
- **Oracle cost:** ~ms (verify by testing all 2^N input patterns for small N, or random sampling for large N)
- **Why interesting:** combinatorial, provably optimal solutions exist for small N,
  open problems for large N, clean mathematical structure
- **Action:** agent writes a function returning a list of comparator pairs

### Boolean circuit minimization
Given a truth table or boolean function spec, produce a circuit (AND/OR/NOT/XOR gates)
with minimal gate count that implements it.
- **Metric:** gate count (or weighted gate cost, or circuit depth)
- **Oracle cost:** ~ms (evaluate circuit on all inputs or sample)
- **Why interesting:** NP-hard, directly relevant to chip design and compiler
  optimization, can generate random target functions for curriculum
- **Action:** agent writes a circuit description or gate list

### SAT solver heuristics
Write a DPLL/CDCL-style SAT solver, optimizing decisions and restarts on a benchmark suite.
- **Metric:** total decisions + conflicts across benchmark instances (or solve time)
- **Oracle cost:** ~seconds (run solver on benchmark suite)
- **Why interesting:** meta-optimization — writing code that searches. Exercises
  fundamentally different reasoning than direct optimization. Can use standard
  benchmark suites (SATLIB) for evaluation.
- **Action:** agent writes the variable selection heuristic, restart policy, and clause learning strategy

### Bin packing (2D)
Pack a set of rectangles into a strip of fixed width, minimizing strip height.
- **Metric:** strip height (or wasted area)
- **Oracle cost:** ~ms (place rectangles, compute height)
- **Why interesting:** NP-hard, visually interpretable, many known heuristic
  families (shelf, guillotine, skyline), industrially relevant
- **Action:** agent writes the packing algorithm

### TSP heuristics
Given a set of cities, write a heuristic that produces a short tour.
- **Metric:** tour length / optimal tour length (for small instances with known optima)
- **Oracle cost:** ~ms (compute tour length)
- **Why interesting:** classic optimization, easy to verify, many heuristic
  families (nearest neighbor, 2-opt, Lin-Kernighan), can scale difficulty with city count
- **Action:** agent writes the tour construction + improvement heuristic

### Job-shop scheduling
Schedule N jobs on M machines, each job being a sequence of operations with
machine assignments and durations, minimizing makespan.
- **Metric:** makespan (total completion time)
- **Oracle cost:** ~ms (simulate schedule)
- **Why interesting:** combinatorial, real-world applicable (manufacturing,
  computing), well-studied benchmarks (Taillard, OR-Library)
- **Action:** agent writes the scheduling/dispatching logic

### Matrix multiply tiling
Optimize a matrix multiplication kernel for a simulated cache hierarchy,
minimizing cache misses or simulated cycles.
- **Metric:** simulated cycles or cache misses
- **Oracle cost:** ~seconds (run through cache simulator)
- **Why interesting:** close to kernel-opt but tests different architectural
  reasoning (memory hierarchy vs instruction-level parallelism), directly
  relevant to HPC
- **Action:** agent writes the tiled matmul implementation

### Cache replacement policy
Given a memory access trace, write a cache eviction policy that maximizes hit rate.
- **Metric:** hit rate on trace
- **Oracle cost:** ~ms (replay trace through cache simulator)
- **Why interesting:** systems reasoning, well-studied (LRU, LFU, ARC, etc.),
  Belady's optimal gives an upper bound, can vary traces for curriculum
- **Action:** agent writes the eviction policy (on_access, on_miss, on_evict hooks)

### Compression codec
Design a compression algorithm optimizing for compression ratio and decompression speed
on a given corpus.
- **Metric:** compressed_size * decompression_time (or Pareto metric)
- **Oracle cost:** ~seconds (compress + decompress corpus)
- **Why interesting:** information-theoretic depth, many approach families
  (dictionary, entropy, transform), practical relevance
- **Action:** agent writes the compress/decompress functions

### Regex optimization
Given a set of strings to match and strings to reject, write the fastest regex.
- **Metric:** matching throughput (strings/second) while maintaining correctness
- **Oracle cost:** ~ms (run regex on test corpus)
- **Why interesting:** string processing, well-defined correctness, surprising
  optimization depth (backtracking avoidance, character class optimization, anchoring)
- **Action:** agent writes the regex pattern and optionally a pre-filter

### Database query planning
Given a schema, statistics, and a SQL query, write a physical query plan minimizing
estimated cost on a simulated executor.
- **Metric:** simulated I/O + CPU cost
- **Oracle cost:** ~seconds (run plan on simulator with stats)
- **Why interesting:** multi-step reasoning about data flow, join ordering is
  NP-hard, index selection adds another dimension
- **Action:** agent writes the physical plan (join order, access methods, algorithms)

## Prior art: LLM-driven code evolution systems

These systems use LLMs to evolve code solutions in an iterative evaluate-mutate loop.
The problems they target are directly relevant as candidate RL environments.

### AlphaEvolve (Google DeepMind, 2025)

[Paper](https://arxiv.org/abs/2506.13131) |
[Math problems repo](https://github.com/google-deepmind/alphaevolve_repository_of_problems)

Gemini-powered evolutionary coding agent. Maintains island populations of programs,
uses LLM as mutation/crossover operator. Evaluated on 67 math problems + Google infra.

**Engineering / infrastructure problems:**
- Data center job scheduling (Borg) — recovered 0.7% of Google's worldwide compute
- TPU arithmetic circuit simplification — removed unnecessary bits in Verilog
- Matrix multiply kernel for Gemini training — 23% kernel speedup, 1% training time reduction
- FlashAttention GPU kernel optimization — up to 32.5% speedup

**Math problems (67 total, sampled highlights):**
- Kakeya and Nikodym sets in finite fields
- Autocorrelation problems related to Sidon sets
- Erdos minimum overlap problem
- 3D Kakeya problem
- Gagliardo-Nirenberg inequality bounds
- Sendov's conjecture (polynomial zeros and critical points)
- Crouzeix's conjecture (operator norms over numerical range)
- Arithmetic Kakeya conjecture
- Packing N copies in a dilated shape
- Erdos discrepancy problem
- Sum-product problems in rings
- Max-to-min distance ratios for point configurations
- Erdos squares-in-a-square problem
- No-5-on-a-sphere problem (grid subsets avoiding coplanar/cospherical quintuplets)
- IMO 2025 Problem 6 variant (rectangular tiling)
- Logic puzzles (decision problems with adversarial guards)
- Kissing number in 11 dimensions — new lower bound of 593
- 4x4 complex matrix multiplication — 48 scalar multiplications (first improvement over Strassen in 56 years)

### ShinkaEvolve (Sakana AI, 2025)

[Paper](https://arxiv.org/abs/2509.19349) |
[GitHub](https://github.com/SakanaAI/ShinkaEvolve) |
[Blog](https://sakana.ai/shinka-evolve/)

Open-source evolutionary framework. Key innovations: parent sampling balancing
exploration/exploitation, code novelty rejection-sampling, bandit-based LLM ensemble.

**Problems:**
1. **Circle packing** — pack 26 circles in a unit square, maximize sum of radii.
   New SOTA in ~150 evaluations. Discovered hybrid of golden-angle spiral + gradient
   refinement + simulated annealing.
2. **AIME math reasoning** — evolve agent scaffolds for competition math. Evolved a
   three-stage architecture (expert personas, peer review, synthesis) in 75 generations.
3. **ALE-Bench competitive programming** — improve heuristic solutions for 10 AtCoder
   NP-hard problems (AHC015). +2.3% average, one solution would have placed 2nd.
4. **MoE load balancing loss** — evolve loss functions for mixture-of-experts training.
   Outperformed DeepSeek's Global LBL in 30 generations, validated on 7 benchmarks.

### DeltaEvolve (Jiang et al., 2026)

[Paper](https://arxiv.org/abs/2602.02919)

Momentum-driven evolution using structured semantic deltas instead of full code histories.
Formalizes evolution as EM framework. Matches/beats AlphaEvolve with fewer tokens.

**Problems:**
1. **BBOB black-box optimization** — continuous function minimization (sphere 3D,
   Rosenbrock 5D, Rastrigin 10D, ellipsoid 20D, Schaffer's 40D)
2. **Hexagon packing** — pack 11 unit hexagons in smallest outer hexagon
3. **Symbolic regression** — discover expression for nonlinear harmonic oscillator
4. **PDE solver** — optimize conjugate gradient for 2D Poisson equation
   (50x50, 100x100, 200x200 grids)
5. **Efficient convolution** — optimize 2D convolution vs scipy baseline, discovered
   FFT-based approach autonomously

### ContextEvolve (2026)

[Paper](https://arxiv.org/abs/2602.02597)

Multi-agent context compression for systems code optimization. Decomposes optimization
context into orthogonal dimensions via Summarizer, Navigator, and specialist agents.
Outperforms GEPA and OpenEvolve by 6.5% on average.

**Problems (ADRS benchmark):**
1. **Transaction scheduling** — optimize execution order in a database, minimize makespan
2. **SQL optimization** — reorder table rows/fields to maximize KV cache hit rate
3. **Load balancing** — distribute computation across GPUs in MoE models
4. **Sparse attention kernel** — optimize attention mask density vs sparsity-induced error
5. **Model placement** — place large ML models across GPUs to reduce cache contention

## References

- [CompilerGym](https://github.com/facebookresearch/CompilerGym) — compiler optimization as RL
- [CodeRL](https://github.com/salesforce/CodeRL) — RL with unit test rewards for code generation
- [CodeContests](https://github.com/google-deepmind/code_contests) — competitive programming dataset (AlphaCode)
- [APPS](https://github.com/hendrycks/apps) — 10k coding problems benchmark
- [LiveCodeBench](https://github.com/livecodebench/livecodebench) — continuously updated competitive programming benchmark
- [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) — RL training for SWE-bench tasks
- [MLAgentBench](https://arxiv.org/abs/2310.03302) — agents optimizing ML pipelines
- [AlphaEvolve](https://arxiv.org/abs/2506.13131) — Gemini-powered evolutionary coding agent (DeepMind)
- [AlphaEvolve math problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems) — 67 math problems with code
- [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) — open-source sample-efficient program evolution (Sakana AI)
- [DeltaEvolve](https://arxiv.org/abs/2602.02919) — momentum-driven evolution with semantic deltas
- [ContextEvolve](https://arxiv.org/abs/2602.02597) — multi-agent context compression for systems code optimization
