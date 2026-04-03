"""Autocorrelation Inequality - Naive constant solution."""


def solve():
    """Return a list of 200 non-negative floats (step function heights).

    The step function is defined on 200 equally-spaced intervals on [-1/4, 1/4].
    Goal: minimize max_{|t|<=1/2} (f*f)(t) / (integral f)^2.
    """
    n = 200
    # Naive: constant function f = 1.0 everywhere
    return [1.0] * n
