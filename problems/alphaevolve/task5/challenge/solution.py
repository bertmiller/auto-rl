"""Max-Min Distance Ratio in 2D (n=12) - Naive regular polygon solution."""

import math


def solve():
    """Return 12 points in R^2 minimizing max/min pairwise distance ratio.

    Returns:
        list: A list of 12 (x, y) tuples.
    """
    n = 12
    # Place points on a regular 12-gon (unit circle)
    points = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        points.append((math.cos(angle), math.sin(angle)))
    return points
