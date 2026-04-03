"""Oracle for Task 19: Blue Channel Extraction from RGB."""

import random
import sys
import time

NUM_PIXELS = 50_000_000  # 150 MB input, 50 MB output
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    return rng.randbytes(NUM_PIXELS * 3)


def reference_solve(data):
    return data[2::3]


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    solve(data[:300])

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if not isinstance(result, bytes):
        print(f"WRONG: expected bytes, got {type(result).__name__}", file=sys.stderr)
        sys.exit(1)

    if len(result) != NUM_PIXELS:
        print(f"WRONG: expected {NUM_PIXELS} bytes, got {len(result)}", file=sys.stderr)
        sys.exit(1)

    if result != expected:
        print("WRONG: blue channel mismatch", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
