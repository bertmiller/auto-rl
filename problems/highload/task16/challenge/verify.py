"""Oracle for Task 16: Large Integer Multiplication."""

import random
import sys
import time

N = 5_000  # bytes per integer
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    a_bytes = rng.randbytes(N)
    b_bytes = rng.randbytes(N)
    return a_bytes, b_bytes


def reference_solve(a_bytes, b_bytes):
    a = int.from_bytes(a_bytes, byteorder="little")
    b = int.from_bytes(b_bytes, byteorder="little")
    product = a * b
    result_len = len(a_bytes) + len(b_bytes)
    return product.to_bytes(result_len, byteorder="little")


def main():
    a_bytes, b_bytes = generate_data()
    expected = reference_solve(a_bytes, b_bytes)

    from solution import solve

    # Warmup
    small_a = a_bytes[:50]
    small_b = b_bytes[:50]
    solve(small_a, small_b)

    start = time.perf_counter()
    result = solve(a_bytes, b_bytes)
    elapsed = time.perf_counter() - start

    if not isinstance(result, bytes):
        print(f"WRONG: expected bytes, got {type(result).__name__}", file=sys.stderr)
        sys.exit(1)

    expected_len = len(a_bytes) + len(b_bytes)
    if len(result) != expected_len:
        print(f"WRONG: expected {expected_len} bytes, got {len(result)}", file=sys.stderr)
        sys.exit(1)

    if result != expected:
        print("WRONG: product mismatch", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
