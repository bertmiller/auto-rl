# Sums and Differences of Finite Sets

## Problem

Given a finite set A of distinct non-negative integers with |A| >= 3, compute the sumset A+A and the difference set A-A:

- **Sumset**: A+A = {a + b : a, b in A}
- **Difference set**: A-A = {a - b : a, b in A}

Maximize the ratio: **log(|A+A| / |A|) / log(|A-A| / |A|)**.

## Background

For any finite set A of integers, a classical result in additive combinatorics states that |A+A| and |A-A| are related but can differ significantly depending on the structure of A. Arithmetic progressions have |A+A| ~ 2|A| and |A-A| ~ 2|A|, giving a ratio near 1. Sets with more additive structure (like geometric-style constructions) can have the sumset grow much faster than the difference set relative to |A|.

## Constraints

- A must contain at least 3 distinct non-negative integers
- All elements must be non-negative integers
- Both |A+A|/|A| and |A-A|/|A| must be > 1 (guaranteed if |A| >= 3 with distinct elements)

## Objective

Maximize **log(|A+A| / |A|) / log(|A-A| / |A|)**.

## Interface

Your `solve()` function must return a `list[int]` of distinct non-negative integers (at least 3 elements).

## Scoring

- If the input is invalid (non-integers, fewer than 3 distinct elements, or |A-A|/|A| <= 1), the score is **-1**.
- Otherwise, the score is log(|A+A|/|A|) / log(|A-A|/|A|).

## Notes

- A naive arithmetic progression (e.g., range(50)) gives a ratio close to 1.0.
- Sets with special multiplicative or combinatorial structure can achieve higher ratios.
- The set size matters: larger sets give more room for structure but also more pairs.
