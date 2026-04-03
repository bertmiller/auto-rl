"""Heilbronn Triangle Problem (n=12) - Grid with y-jitter to avoid collinearity."""

import math


def solve():
    """Return 12 points inside the unit equilateral triangle maximizing
    the minimum triangle area over all triples.

    The triangle has vertices at (0,0), (1,0), (0.5, sqrt(3)/2).

    Returns:
        list: List of 12 (x, y) tuples.
    """
    h = math.sqrt(3) / 2  # ~ 0.8660

    # Place 12 points in a 4-row layout (4,3,3,2) with y-jitter
    # so that no row has 3+ points at the exact same y-coordinate.
    # The jitter breaks collinearity within rows, and the staggered
    # x-positions break it across rows.

    base_rows = [
        # (base_y_frac, list of x_fracs within available width)
        (0.15, [0.12, 0.37, 0.63, 0.88]),
        (0.40, [0.20, 0.50, 0.80]),
        (0.65, [0.25, 0.50, 0.75]),
        (0.85, [0.35, 0.65]),
    ]

    # Unique y-jitters for each of 12 points (as fraction of h)
    y_jitters = [
        0.012, -0.018, 0.007, -0.013,
        0.015, -0.009, 0.019,
        -0.016, 0.011, -0.014,
        0.008, -0.006,
    ]

    points = []
    jitter_idx = 0
    for base_y_frac, x_fracs in base_rows:
        for xf in x_fracs:
            y = (base_y_frac + y_jitters[jitter_idx]) * h
            y = max(0.01, min(h - 0.01, y))

            frac = y / h
            x_left = 0.5 * frac
            x_right = 1.0 - 0.5 * frac
            width = x_right - x_left

            x = x_left + xf * width
            x = max(x_left + 0.005, min(x_right - 0.005, x))

            points.append((x, y))
            jitter_idx += 1

    return points
