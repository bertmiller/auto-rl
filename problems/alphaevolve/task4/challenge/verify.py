"""Verify and score a Heilbronn triangle problem solution for n=12."""

import numpy as np
import math
import sys

N = 12
# Unit equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, math.sqrt(3) / 2])

TOL = 1e-9


def point_in_triangle(p, v0, v1, v2):
    """Check if point p is inside triangle (v0, v1, v2) using barycentric coords."""
    d00 = np.dot(v1 - v0, v1 - v0)
    d01 = np.dot(v1 - v0, v2 - v0)
    d02 = np.dot(v1 - v0, p - v0)
    d11 = np.dot(v2 - v0, v2 - v0)
    d12 = np.dot(v2 - v0, p - v0)

    inv_denom = 1.0 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * inv_denom
    v = (d00 * d12 - d01 * d02) * inv_denom

    return (u >= -TOL) and (v >= -TOL) and (u + v <= 1.0 + TOL)


def triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) +
                      p2[0] * (p3[1] - p1[1]) +
                      p3[0] * (p1[1] - p2[1]))


def verify_and_score():
    from solution import solve

    points = solve()

    # Validate return format
    if not isinstance(points, (list, tuple)) or len(points) != N:
        print("score=-1")
        return

    pts = np.array(points, dtype=np.float64)

    if pts.shape != (N, 2):
        print("score=-1")
        return

    # Check for invalid values
    if not np.all(np.isfinite(pts)):
        print("score=-1")
        return

    # Check all points are inside the triangle
    for i in range(N):
        if not point_in_triangle(pts[i], V0, V1, V2):
            print("score=-1")
            return

    # Compute minimum triangle area over all C(n,3) triples
    min_area = float("inf")
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                area = triangle_area(pts[i], pts[j], pts[k])
                if area < min_area:
                    min_area = area

    score = float(min_area)
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
