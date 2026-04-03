"""Oracle for Task 13: Count uint8."""

import random
import sys
import time

N = 200_000_000
SEED = 42
TARGET = 127


def generate_data():
    rng = random.Random(SEED)
    return rng.randbytes(N)


def reference_solve(data):
    count = 0
    for b in data:
        if b == TARGET:
            count += 1
    return count


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    solve(data[:1000])

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
