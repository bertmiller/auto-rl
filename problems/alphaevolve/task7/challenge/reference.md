# Squares in a Square (n=10)

## Problem

Pack 10 non-overlapping axis-aligned squares inside the unit square [0,1] x [0,1] to maximize the sum of their side lengths.

## Constraints

- You must return exactly 10 squares, each specified as (x, y, side_length).
- (x, y) is the bottom-left corner of the square.
- Each square must be fully inside [0,1]^2:
  - x >= 0, y >= 0
  - x + side_length <= 1, y + side_length <= 1
- side_length > 0 for each square.
- No two squares may overlap (their interiors must be disjoint). Two squares that share only an edge or corner are allowed.

## Objective

Maximize the **sum of side lengths**: sum(s_i for i in 1..10).

## Interface

Your `solve()` function must return a list of 10 `(x, y, side_length)` tuples (floats).

## Scoring

- If any constraint is violated (out of bounds, non-positive side, or overlap), the score is **-1** (invalid).
- Otherwise, the score is the sum of all side lengths.

## Notes

- A naive solution placing 10 equal squares in a 2x5 grid gives side length 0.2 each, for a total of 2.0.
- Better packings use squares of different sizes to fill the space more efficiently.
- This is related to squaring the square problems in combinatorial geometry.
- The overlap check uses axis-aligned bounding boxes: two squares (x1,y1,s1) and (x2,y2,s2) overlap if and only if x1 < x2+s2 and x2 < x1+s1 and y1 < y2+s2 and y2 < y1+s1.
