"""Oracle for Task 18: Matrix Multiplication."""

import random
import sys
import time

N = 350
SEED = 42
TOL = 1e-6


def generate_data():
    rng = random.Random(SEED)
    a = [[rng.uniform(-10, 10) for _ in range(N)] for _ in range(N)]
    b = [[rng.uniform(-10, 10) for _ in range(N)] for _ in range(N)]
    return a, b


def reference_solve(a, b):
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def main():
    a, b = generate_data()
    expected = reference_solve(a, b)

    from solution import solve

    # Warmup with small matrices
    small_a = [row[:3] for row in a[:3]]
    small_b = [row[:3] for row in b[:3]]
    solve(small_a, small_b)

    start = time.perf_counter()
    result = solve(a, b)
    elapsed = time.perf_counter() - start

    if not isinstance(result, list) or len(result) != N:
        print(f"WRONG: expected {N}x{N} matrix", file=sys.stderr)
        sys.exit(1)

    for i in range(N):
        if not isinstance(result[i], list) or len(result[i]) != N:
            print(f"WRONG: row {i} has wrong length", file=sys.stderr)
            sys.exit(1)
        for j in range(N):
            if abs(result[i][j] - expected[i][j]) > TOL:
                print(
                    f"WRONG: element [{i}][{j}] = {result[i][j]}, expected {expected[i][j]}",
                    file=sys.stderr,
                )
                sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
