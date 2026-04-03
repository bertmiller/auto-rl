"""Verify and score a Tammes problem solution for n=30."""

import numpy as np
import sys

N = 30


def verify_and_score():
    from solution import solve

    points = solve()

    # Validate return format
    if not isinstance(points, (list, tuple)) or len(points) != N:
        print("score=-1")
        return

    pts = np.array(points, dtype=np.float64)

    if pts.shape != (N, 3):
        print("score=-1")
        return

    # Check for invalid values
    if not np.all(np.isfinite(pts)):
        print("score=-1")
        return

    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        print("score=-1")
        return
    pts = pts / norms

    # Compute all pairwise Euclidean distances
    min_dist = float("inf")
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d

    score = float(min_dist)
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
