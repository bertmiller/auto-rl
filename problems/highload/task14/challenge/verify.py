"""Oracle for Task 14: Blue from RGBA."""

import random
import sys
import time

N_PIXELS = 60_000_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    return rng.randbytes(N_PIXELS * 4)


def reference_solve(data):
    return bytes(data[i] for i in range(2, len(data), 4))


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    solve(data[:400])

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got length {len(result)}, expected length {len(expected)}", file=sys.stderr)
        if len(result) == len(expected):
            for i in range(len(result)):
                if result[i] != expected[i]:
                    print(f"First mismatch at pixel {i}: got {result[i]}, expected {expected[i]}", file=sys.stderr)
                    break
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
