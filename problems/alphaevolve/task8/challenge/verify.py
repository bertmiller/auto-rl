"""Verify and score a block stacking overhang solution for n=20 blocks."""

import sys

N = 20
TOL = 1e-9
INVALID_SCORE = -1


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)) or len(result) != N:
        print(f"score={INVALID_SCORE}")
        return

    x = [float(v) for v in result]

    # Check stability: for each block i (0..N-2), the center of mass of
    # blocks i+1..N-1 must lie within [x[i], x[i]+1]
    for i in range(N - 1):
        com = 0.0
        count = 0
        for j in range(i + 1, N):
            com += x[j] + 0.5  # center of block j
            count += 1
        com /= count

        if com < x[i] - TOL or com > x[i] + 1.0 + TOL:
            print(f"score={INVALID_SCORE}")
            return

    # Check table support: center of mass of ALL blocks must be <= 0
    total_com = sum(x[j] + 0.5 for j in range(N)) / N
    if total_com > 0.0 + TOL:
        print(f"score={INVALID_SCORE}")
        return

    # Score: rightmost extent of any block
    score = max(x[i] + 1.0 for i in range(N))
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
