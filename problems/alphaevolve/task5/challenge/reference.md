# Max-Min Distance Ratio in 2D (n=12)

## Problem

Place 12 points in the plane R^2 to minimize the ratio of the maximum pairwise distance to the minimum pairwise distance.

## Constraints

- You must return exactly 12 points as (x, y) tuples.
- No two points may be coincident (minimum pairwise distance must be > 1e-12).

## Objective

Minimize max_dist / min_dist, where max_dist and min_dist are taken over all pairs of distinct points.

Since the optimization framework maximizes, the score is the **negative** of this ratio: score = -(max_dist / min_dist). A higher (less negative) score is better.

## Interface

Your `solve()` function must return a list of 12 `(x, y)` tuples (floats).

## Scoring

- If fewer than 12 points are returned, or any two points coincide, the score is **-1e10** (invalid).
- Otherwise, the score is -(max_pairwise_distance / min_pairwise_distance).

## Notes

- A regular 12-gon on a circle achieves a ratio of 1/sin(pi/12) ~ 3.864, giving a score of about -3.864.
- The optimal arrangement for minimizing this ratio is related to optimal point configurations in combinatorial geometry.
- Scale and translation don't matter since the ratio is scale-invariant; only the shape matters.
