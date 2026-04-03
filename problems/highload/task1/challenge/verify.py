"""Oracle for Task 1: Sum of Numbers (Parse Integers)."""

import random
import sys
import time

MASK = 0xFFFFFFFFFFFFFFFF
N = 30_000_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    return [str(rng.randint(0, 2**31 - 1)) for _ in range(N)]


def reference_solve(numbers):
    total = 0
    for s in numbers:
        total += int(s)
    return total & MASK


def main():
    numbers = generate_data()
    expected = reference_solve(numbers)

    from solution import solve

    # Warmup
    solve(numbers[:100])

    start = time.perf_counter()
    result = solve(numbers)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
