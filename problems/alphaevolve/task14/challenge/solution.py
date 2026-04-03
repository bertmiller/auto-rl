"""Erdos Minimum Overlap - Naive constant solution."""


def solve():
    """Return a list of 200 floats in [0, 1] representing h on [0, 2].

    The function h should satisfy integral_0^2 h(x) dx = 1 (mean = 0.5).
    Goal: minimize the maximum overlap integral h(x)(1 - h(x+k)) dx over shifts k.
    """
    n = 200
    # Naive: constant function h = 0.5 everywhere
    return [0.5] * n
