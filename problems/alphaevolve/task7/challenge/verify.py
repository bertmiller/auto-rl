"""Verify and score a squares-in-a-square packing solution for n=10."""

import sys

N = 10
TOL = 1e-9
INVALID_SCORE = -1


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)) or len(result) != N:
        print(f"score={INVALID_SCORE}")
        return

    squares = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            print(f"score={INVALID_SCORE}")
            return
        x, y, s = float(item[0]), float(item[1]), float(item[2])
        squares.append((x, y, s))

    # Check each square is valid and inside [0,1]^2
    for x, y, s in squares:
        if s <= 0:
            print(f"score={INVALID_SCORE}")
            return
        if x < -TOL or y < -TOL:
            print(f"score={INVALID_SCORE}")
            return
        if x + s > 1.0 + TOL or y + s > 1.0 + TOL:
            print(f"score={INVALID_SCORE}")
            return

    # Check no overlap between any pair of squares
    # Two axis-aligned squares overlap if their interiors intersect
    for i in range(N):
        x1, y1, s1 = squares[i]
        for j in range(i + 1, N):
            x2, y2, s2 = squares[j]
            # Overlap iff projections overlap on both axes
            # Interior overlap: strict inequalities
            if (x1 < x2 + s2 - TOL and x2 < x1 + s1 - TOL and
                    y1 < y2 + s2 - TOL and y2 < y1 + s1 - TOL):
                print(f"score={INVALID_SCORE}")
                return

    score = sum(s for _, _, s in squares)
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
