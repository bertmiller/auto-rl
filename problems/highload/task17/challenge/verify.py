"""Oracle for Task 17: MD5 Hash."""

import hashlib
import random
import sys
import time

N = 10_000_000  # 10 MB
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    return rng.randbytes(N)


def reference_solve(data):
    return hashlib.md5(data).hexdigest()


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    solve(data[:1000])

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if not isinstance(result, str):
        print(f"WRONG: expected str, got {type(result).__name__}", file=sys.stderr)
        sys.exit(1)

    if len(result) != 32:
        print(f"WRONG: expected 32-char hex string, got length {len(result)}", file=sys.stderr)
        sys.exit(1)

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
