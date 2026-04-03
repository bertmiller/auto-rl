"""Verify and score a Thomson problem solution for n=30."""

import numpy as np
import sys

N = 30
INVALID_SCORE = -1e10


def verify_and_score():
    from solution import solve

    points = solve()

    # Validate return format
    if not isinstance(points, (list, tuple)) or len(points) != N:
        print(f"score={INVALID_SCORE:.6e}")
        return

    pts = np.array(points, dtype=np.float64)

    if pts.shape != (N, 3):
        print(f"score={INVALID_SCORE:.6e}")
        return

    # Check for invalid values
    if not np.all(np.isfinite(pts)):
        print(f"score={INVALID_SCORE:.6e}")
        return

    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        print(f"score={INVALID_SCORE:.6e}")
        return
    pts = pts / norms

    # Compute Coulomb energy: E = sum_{i<j} 1/dist(i,j)
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                # Coincident points
                print(f"score={INVALID_SCORE:.6e}")
                return
            energy += 1.0 / d

    # Return negative energy (higher = better = lower energy)
    score = -energy
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
