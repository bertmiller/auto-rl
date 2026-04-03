# Autocorrelation Inequality (Upper Bound)

## Problem

Find the smallest constant C_1 such that for all non-negative functions f:

    max_{|t| <= 1/2} (f * f)(t) >= C_1 * (integral_{-1/4}^{1/4} f(x) dx)^2

where (f * f)(t) = integral f(t - x) f(x) dx is the autocorrelation of f.

To find a tight upper bound on C_1, we want to construct a function f that **minimizes** the ratio:

    max_{|t| <= 1/2} (f * f)(t) / (integral_{-1/4}^{1/4} f(x) dx)^2

## Approach

We represent f as a step function on [-1/4, 1/4] with n = 200 equally-spaced intervals, each of width dx = 0.5 / 200 = 0.0025. The function is zero outside [-1/4, 1/4].

## Interface

`solve()` returns a `list[float]` of 200 non-negative values representing the heights of the step function on the 200 intervals covering [-1/4, 1/4].

## Scoring

The verifier:
1. Clamps all values to be >= 0
2. Computes the autocorrelation (f * f)(t) via numpy convolution
3. Finds max_{|t| <= 1/2} (f * f)(t) and (integral f)^2
4. Returns score = -(max_autocorrelation / integral_squared)

We **maximize** the score, which means **minimizing** the ratio (finding a tighter bound).

- If all values are zero (integral ~ 0), returns score = -1e10 (invalid).
- A constant function (all heights = 1.0) gives a baseline score around -2.0.

## Notes

- The theoretical minimum ratio involves clever constructions that concentrate mass in specific patterns.
- Consider functions that are zero in parts of the interval, or have varying heights.
- The autocorrelation at t=0 always equals integral(f^2), so the max is at least that large.
