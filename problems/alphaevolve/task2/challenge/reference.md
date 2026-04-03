# Tammes Problem (n=30)

## Problem

Place 30 points on the surface of the unit sphere such that the minimum pairwise Euclidean distance is maximized.

This is the classical Tammes problem (or spherical packing problem): distribute points on a sphere so they are as spread out as possible.

## Constraints

- You must return exactly 30 points.
- Each point (x, y, z) will be normalized to the unit sphere by the verifier.
- Points must be finite (no NaN or Inf).

## Objective

Maximize the **minimum pairwise Euclidean distance** among all pairs of points.

## Interface

Your `solve()` function must return a list of 30 `(x, y, z)` tuples.

## Scoring

- If fewer than 30 points are returned or any point is invalid, the score is **-1**.
- Otherwise, the score is the minimum Euclidean distance over all C(30,2) = 435 pairs.

## Notes

- A Fibonacci sphere gives a reasonable starting point with min distance around 0.56.
- The theoretical best for 30 points is approximately 0.647 (min distance).
- Consider iterative repulsion methods, simulated annealing, or gradient-based optimization.
- The verifier normalizes all points to the unit sphere, so you only need to specify directions.
