"""Oracle for Task 20: Maximum Square."""

import random
import sys
import time

ROWS = 5000
COLS = 5000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    matrix = []
    for _ in range(ROWS):
        row = [rng.randint(0, 1) for _ in range(COLS)]
        matrix.append(row)
    return matrix


def reference_solve(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_side = 0
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                if dp[i][j] > max_side:
                    max_side = dp[i][j]
    return max_side if max_side > 1 else 0


def main():
    matrix = generate_data()
    expected = reference_solve(matrix)

    from solution import solve

    # Warmup
    small = [[1, 1], [1, 1]]
    solve(small)

    start = time.perf_counter()
    result = solve(matrix)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
