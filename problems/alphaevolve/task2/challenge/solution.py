"""Tammes Problem (n=30) - Fibonacci sphere naive solution."""

import math


def solve():
    """Return 30 points on the unit sphere maximizing min pairwise distance.

    Returns:
        list: List of 30 (x, y, z) tuples on the unit sphere.
    """
    n = 30
    golden_ratio = (1 + math.sqrt(5)) / 2
    points = []

    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))

    return points
