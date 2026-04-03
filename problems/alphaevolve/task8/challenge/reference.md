# Block Stacking Overhang (n=20)

## Problem

Stack 20 identical blocks (each of width 1) to maximize the total overhang from a table edge.

## Setup

- A table occupies x <= 0 (the right edge of the table is at x=0).
- You return 20 x-positions: x[0], x[1], ..., x[19], where x[i] is the **left edge** of block i.
- Block 0 is the bottom block (on the table), block 19 is the top.
- Each block occupies the interval [x[i], x[i] + 1].
- Block i rests on block i-1 (for i >= 1).

## Constraints

- **Stability (block-on-block)**: for each i in {0, 1, ..., 18}, the center of mass of blocks i+1, i+2, ..., 19 must lie within [x[i], x[i] + 1]. The center of block j is x[j] + 0.5.
- **Table support**: the center of mass of ALL 20 blocks must satisfy COM <= 0 (i.e., the combined center of mass must be on or to the left of the table edge).

## Objective

Maximize the **rightmost extent** of any block: score = max(x[i] + 1 for all i).

## Interface

Your `solve()` function must return a list of 20 floats (the x-positions of the left edges).

## Scoring

- If the number of positions is not 20, the score is **-1** (invalid).
- If any stability constraint is violated, the score is **-1** (invalid).
- If the table support constraint is violated, the score is **-1** (invalid).
- Otherwise, the score is max(x[i] + 1 for i in 0..19).

## Notes

- In the classic optimal single-stack overhang, blocks are positioned so each sub-stack's center of mass is at the right edge of the block below.
- The optimal offsets between consecutive blocks (from top down) follow the harmonic series: 1/2, 1/4, 1/6, ...
- The maximum overhang for n blocks in a single stack is H(n)/2 where H(n) = 1 + 1/2 + ... + 1/n is the n-th harmonic number.
- For n=20: H(20)/2 ~ 1.799, so the rightmost extent of the top block is about 1.799.
- The bottom block must be positioned far enough left that the total center of mass stays on the table.
