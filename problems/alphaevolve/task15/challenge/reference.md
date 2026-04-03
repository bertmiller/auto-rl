# Difference Bases

## Problem

A **difference basis** for n is a set B of non-negative integers such that every integer in {1, 2, ..., n} can be expressed as a difference a - b where a, b are in B and a > b.

The goal is to find a set B that is a difference basis for the largest possible n, while keeping |B| small.

## Metric

The score is:

    n / |B|^2

where n is the largest integer such that {1, ..., n} is contained in the set of positive differences of B, and |B| is the size of the set.

It is known that the optimal ratio approaches a constant as n grows (related to perfect difference sets and Singer difference sets).

## Interface

`solve()` returns a `list[int]` of distinct non-negative integers forming a difference basis.

## Scoring

The verifier:
1. Takes the set B and computes all positive differences {a - b : a, b in B, a > b}
2. Finds the largest n such that {1, 2, ..., n} is a subset of the differences
3. Returns n / len(B)^2 as the score

- If B has fewer than 2 elements, returns score = -1.
- If no consecutive range {1, ..., n} starting from 1 exists, returns score = -1.

## Notes

- Naive: B = {0, 1, 2, ..., k} covers {1, ..., k} with |B| = k+1, giving score = k/(k+1)^2 ~ 1/k.
- Better constructions use structures from combinatorics (e.g., quadratic residues, Singer difference sets).
- A perfect difference set of order q (where q is a prime power) has |B| = q+1 elements and covers {1, ..., q^2+q}, giving ratio (q^2+q)/(q+1)^2 ~ 1.
- Example: for q=7, B could be based on {0, 1, 2, 4} mod 7 type constructions scaled appropriately.
- Larger constructions with higher q values will score better if they achieve ratios closer to 1.
