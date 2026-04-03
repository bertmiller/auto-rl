"""Difference Bases - Naive consecutive solution."""


def solve():
    """Return a list of distinct non-negative integers forming a difference basis.

    Goal: maximize n / |B|^2 where n is the largest integer such that
    {1, ..., n} is contained in the positive differences of B.
    """
    # Naive: B = {0, 1, 2, ..., 20}
    # Covers {1, ..., 20} with |B| = 21, score = 20/441 ~ 0.0454
    return list(range(21))
