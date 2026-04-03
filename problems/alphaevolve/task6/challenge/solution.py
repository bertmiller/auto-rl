"""Convex Hull Volume on Unit Sphere (n=12) - Icosahedron vertices."""

import math


def solve():
    """Return 12 points on the unit sphere maximizing convex hull volume.

    Returns:
        list: A list of 12 (x, y, z) tuples.
    """
    # Regular icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2  # golden ratio

    vertices = [
        (0, 1, phi),
        (0, 1, -phi),
        (0, -1, phi),
        (0, -1, -phi),
        (1, phi, 0),
        (1, -phi, 0),
        (-1, phi, 0),
        (-1, -phi, 0),
        (phi, 0, 1),
        (phi, 0, -1),
        (-phi, 0, 1),
        (-phi, 0, -1),
    ]

    # Normalize to unit sphere
    result = []
    for x, y, z in vertices:
        norm = math.sqrt(x * x + y * y + z * z)
        result.append((x / norm, y / norm, z / norm))

    return result
