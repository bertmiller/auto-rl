# Convex Hull Volume on the Unit Sphere (n=12)

## Problem

Place 12 points on the unit sphere in R^3 to maximize the volume of their convex hull.

## Constraints

- You must return exactly 12 points as (x, y, z) tuples.
- Points are automatically normalized to the unit sphere (projected onto ||p|| = 1).

## Objective

Maximize the **volume** of the convex hull of the 12 points on the unit sphere.

## Interface

Your `solve()` function must return a list of 12 `(x, y, z)` tuples (floats). Each point will be normalized to lie on the unit sphere before scoring.

## Scoring

- If fewer than 12 points are returned, the score is **-1** (invalid).
- If any point is the zero vector (cannot be normalized), the score is **-1**.
- If the convex hull computation fails (e.g., degenerate configuration), the score is **-1**.
- Otherwise, the score is the volume of the convex hull.

## Notes

- The regular icosahedron has 12 vertices and is known to be very close to optimal for maximizing convex hull volume with 12 points on a sphere.
- The maximum possible volume for any convex body inscribed in the unit sphere is (4/3)*pi ~ 4.189, but no polyhedron with 12 vertices achieves this.
- The icosahedron volume is 5*(3+sqrt(5))/12 ~ 2.182.
- Small perturbations from the icosahedron may or may not improve the volume.
