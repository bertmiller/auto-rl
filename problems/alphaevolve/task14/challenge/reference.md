# Erdos Minimum Overlap Problem

## Problem

Find the smallest constant C_5 such that for all measurable f, g: [-1, 1] -> [0, 1] with f + g = 1 and integral(f) = 1:

    sup_{x in [-2, 2]} integral_{-1}^{1} f(t) g(x + t) dt >= C_5

This is equivalent to: given a function h: [0, 2] -> [0, 1] with integral_0^2 h(x) dx = 1, find:

    min_h max_k integral h(x) (1 - h(x + k)) dx

where the max is over shifts k.

## Approach

We represent h as a step function on [0, 2] with n = 200 equally-spaced intervals, each of width dx = 2/200 = 0.01. The values must be in [0, 1] and the integral should equal 1 (mean = 0.5). The verifier normalizes h to satisfy the integral constraint.

## Interface

`solve()` returns a `list[float]` of 200 values in [0, 1] representing h on the 200 intervals covering [0, 2].

## Scoring

The verifier:
1. Clamps values to [0, 1]
2. Normalizes so the integral equals 1 (scales values so mean = 0.5)
3. For each discrete shift k, computes integral h(x)(1 - h(x+k)) dx
4. Takes the maximum over all shifts k
5. Returns score = -(max overlap)

We **maximize** the score, which means **minimizing** the maximum overlap (finding a tighter bound).

- A constant h = 0.5 everywhere gives a baseline.
- The theoretical optimal is related to the Erdos minimum overlap constant, approximately 0.3558.

## Notes

- Think about functions that have low overlap with their own shifts.
- Binary-valued functions (h in {0, 1}) correspond to characteristic functions of sets.
- The problem is related to the Littlewood conjecture and additive combinatorics.
