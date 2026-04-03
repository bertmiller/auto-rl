"""Verify and score a max-min distance ratio solution for n=12 points in 2D."""

import math
import sys

N = 12
COINCIDENCE_TOL = 1e-12
INVALID_SCORE = -1e10


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)) or len(result) != N:
        print(f"score={INVALID_SCORE}")
        return

    points = []
    for p in result:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            print(f"score={INVALID_SCORE}")
            return
        points.append((float(p[0]), float(p[1])))

    # Compute all pairwise distances
    min_dist = float("inf")
    max_dist = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < COINCIDENCE_TOL:
                print(f"score={INVALID_SCORE}")
                return
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    ratio = max_dist / min_dist
    score = -ratio
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
