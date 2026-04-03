# No Isosceles Triangles on Grid (n=15)

## Problem

Place as many points as possible on a 15x15 integer grid such that no three points form an isosceles triangle.

## Definitions

A set of points on the grid [0, 14]^2 is **isosceles-free** if for every triple of distinct points {a, b, c}, all three pairwise distances are distinct:

    d(a,b) != d(a,c)  AND  d(a,b) != d(b,c)  AND  d(a,c) != d(b,c)

where d(p,q) is the Euclidean distance between points p and q.

Equivalently, using squared distances (to avoid floating point issues): for every triple {a, b, c}, the three values d^2(a,b), d^2(a,c), d^2(b,c) must all be distinct.

## Constraints

- All points must have integer coordinates in [0, 14] x [0, 14].
- All points must be distinct.
- No three points may form an isosceles triangle (all three pairwise distances must be distinct).

## Objective

Maximize the **number of points** placed.

## Interface

Your `solve()` function must return a `list[tuple[int, int]]` of grid points.

## Scoring

- If any coordinate is outside [0, 14], or points are not distinct, or any isosceles triple exists, the score is **-1**.
- Otherwise, the score is the number of points.

## Notes

- The grid has 15 x 15 = 225 points total.
- A greedy approach (adding random points and checking) can find 10-15 points easily.
- Sophisticated constructions or search algorithms can find 20-40+ points.
- The constraint grows as O(k^3) in the number of points k, making large solutions hard.
- Using squared distances avoids floating-point precision issues entirely.
