"""Verify and score a convex hull volume solution for n=12 points on the unit sphere."""

import math
import sys

N = 12
INVALID_SCORE = -1
NORM_TOL = 1e-12


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)) or len(result) != N:
        print(f"score={INVALID_SCORE}")
        return

    points = []
    for p in result:
        if not isinstance(p, (list, tuple)) or len(p) != 3:
            print(f"score={INVALID_SCORE}")
            return
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < NORM_TOL:
            print(f"score={INVALID_SCORE}")
            return
        # Normalize to unit sphere
        points.append((x / norm, y / norm, z / norm))

    # Compute convex hull volume using scipy
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        volume = hull.volume
    except Exception:
        print(f"score={INVALID_SCORE}")
        return

    print(f"score={volume:.6f}")


if __name__ == "__main__":
    verify_and_score()
