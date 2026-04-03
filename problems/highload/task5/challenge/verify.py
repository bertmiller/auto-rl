"""Oracle for Task 5: Median."""

import random
import struct
import sys
import time

N = 10_000_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    values = [rng.randint(0, 2**32 - 1) for _ in range(N)]
    data = struct.pack(f"<{N}I", *values)
    return data, values


def reference_solve(values):
    sorted_vals = sorted(values)
    return sorted_vals[len(sorted_vals) // 2]


def main():
    data, values = generate_data()
    expected = reference_solve(values)

    from solution import solve

    # Warmup with small data
    small_data = struct.pack("<5I", 5, 3, 1, 4, 2)
    solve(small_data)

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
