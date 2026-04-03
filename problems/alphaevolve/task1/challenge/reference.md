# Circle Packing in a Unit Square (n=15)

## Problem

Pack 15 non-overlapping circles inside the unit square [0,1] x [0,1] to maximize the sum of their radii.

## Constraints

- You must place exactly 15 circles.
- Each circle is defined by a center (x, y) and radius r > 0.
- Every circle must be fully contained in the unit square:
  - x - r >= 0
  - x + r <= 1
  - y - r >= 0
  - y + r <= 1
- No two circles may overlap: for all pairs i != j, the Euclidean distance between centers must be at least r_i + r_j.

## Objective

Maximize the **sum of all radii**: sum(r_i for i in 1..15).

## Interface

Your `solve()` function must return a tuple `(centers, radii)` where:
- `centers` is a list of 15 `(x, y)` tuples (floats in [0, 1])
- `radii` is a list of 15 positive floats

## Scoring

- If any constraint is violated, the score is **-1** (invalid).
- Otherwise, the score is the sum of radii.

## Notes

- A naive grid layout scores around 0.75. Good packings can exceed 1.2.
- Consider using circles of different sizes to fill gaps efficiently.
- The problem is NP-hard in general; heuristic and iterative approaches work well.
