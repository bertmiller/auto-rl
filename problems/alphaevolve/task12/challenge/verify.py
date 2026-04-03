"""Verify and score a no-isosceles-triangle grid construction (n=15)."""

import sys

GRID_SIZE = 15


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    if len(result) == 0:
        print("score=-1")
        return

    # Parse and validate points
    points = []
    for p in result:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            print("score=-1")
            return
        x, y = int(p[0]), int(p[1])
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            print("score=-1")
            return
        points.append((x, y))

    # Check all points are distinct
    if len(set(points)) != len(points):
        print("score=-1")
        return

    k = len(points)

    # Check no isosceles triple using squared distances
    for i in range(k):
        for j in range(i + 1, k):
            d_ij = (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2
            for l in range(j + 1, k):
                d_il = (points[i][0] - points[l][0]) ** 2 + (points[i][1] - points[l][1]) ** 2
                d_jl = (points[j][0] - points[l][0]) ** 2 + (points[j][1] - points[l][1]) ** 2
                if d_ij == d_il or d_ij == d_jl or d_il == d_jl:
                    print("score=-1")
                    return

    print(f"score={k}")


if __name__ == "__main__":
    verify_and_score()
