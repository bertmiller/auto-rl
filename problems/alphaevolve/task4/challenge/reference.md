# Heilbronn Triangle Problem (n=12)

## Problem

Place 12 points inside the unit equilateral triangle to maximize the minimum area triangle formed by any triple of the 12 points.

The unit equilateral triangle has vertices at:
- (0, 0)
- (1, 0)
- (0.5, sqrt(3)/2) ~ (0.5, 0.8660)

## Constraints

- You must return exactly 12 points.
- Each point (x, y) must be inside or on the boundary of the unit equilateral triangle.
- Points are 2D tuples of floats.

## Objective

Maximize the **minimum triangle area** over all C(12, 3) = 220 triples of points.

## Interface

Your `solve()` function must return a list of 12 `(x, y)` tuples.

## Scoring

- If fewer than 12 points or any point is outside the triangle: score = **-1**.
- Otherwise: score = minimum area over all 220 triples.

## Notes

- The area of a triangle with vertices (x1,y1), (x2,y2), (x3,y3) is:
  0.5 * |x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)|
- A naive grid placement scores around 0.003-0.005.
- Good solutions can achieve minimum area > 0.01.
- Spreading points evenly is key, but the equilateral triangle geometry matters.
- Consider symmetric patterns and iterative improvement.
