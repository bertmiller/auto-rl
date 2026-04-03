# Erdos Discrepancy (D=3)

## Problem

Find the longest possible sequence of +1 and -1 values such that the discrepancy is at most 3.

The **discrepancy** of a sequence x[1], x[2], ..., x[N] is defined as:

    disc(x) = max over all d >= 1, n >= 1 of |sum_{k=1}^{n} x[k*d]|

where the sum is taken over homogeneous arithmetic progressions with common difference d.

In other words, for every step size d and every prefix length n (such that n*d <= N), compute the partial sum of x[d], x[2d], x[3d], ..., x[n*d]. The discrepancy is the maximum absolute value of any such partial sum.

## Background

The Erdos Discrepancy Problem, proven by Terence Tao in 2015, states that for any infinite +/-1 sequence, the discrepancy is unbounded. However, for any fixed bound D, there exist finite sequences with discrepancy <= D. The challenge is to find the longest such sequence.

For D=2, the longest known sequence has length 1160. For D=3, significantly longer sequences are possible.

## Constraints

- Every element must be +1 or -1
- The discrepancy of the sequence must be at most 3
- Maximum sequence length is 5000 (for computational feasibility)

## Objective

Maximize the **length** of the sequence (subject to discrepancy <= 3).

## Interface

Your `solve()` function must return a `list[int]` where every element is +1 or -1.

## Scoring

- If any element is not +1 or -1, the score is **-1**.
- If the discrepancy exceeds 3, the score is **-1**.
- Otherwise, the score is the length of the sequence.

## Notes

- A simple alternating +1, -1, +1, -1, ... sequence has discrepancy 1 and can be made arbitrarily long up to the 5000 cap.
- The real challenge is that the baseline already achieves length 100; you need significantly longer sequences.
- Consider quasi-random or number-theoretic constructions (e.g., character-based sequences).
- For D=3, lengths of several thousand are achievable.
