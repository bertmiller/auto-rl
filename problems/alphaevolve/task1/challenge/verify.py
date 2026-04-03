"""Verify and score a circle packing solution for n=15 in the unit square."""

import numpy as np
import sys

N = 15
TOL = 1e-9


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        print("score=-1")
        return

    centers, radii = result

    if len(centers) != N or len(radii) != N:
        print("score=-1")
        return

    centers = np.array(centers, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)

    # Check radii are positive
    if np.any(radii <= 0):
        print("score=-1")
        return

    # Check circles are inside the unit square
    if np.any(centers[:, 0] - radii < -TOL):
        print("score=-1")
        return
    if np.any(centers[:, 0] + radii > 1.0 + TOL):
        print("score=-1")
        return
    if np.any(centers[:, 1] - radii < -TOL):
        print("score=-1")
        return
    if np.any(centers[:, 1] + radii > 1.0 + TOL):
        print("score=-1")
        return

    # Check no overlap: dist(i,j) >= r_i + r_j for all pairs
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = radii[i] + radii[j]
            if dist < min_dist - TOL:
                print("score=-1")
                return

    score = float(np.sum(radii))
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
