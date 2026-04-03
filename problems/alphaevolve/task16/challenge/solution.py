"""Factoring 20! into 20 Numbers - Naive solution."""

import math


def solve():
    """Return a list of exactly 20 positive integers whose product equals 20!.

    Goal: maximize the minimum of the 20 factors.
    """
    N = 20
    factorial_N = math.factorial(N)

    # Naive: put everything into one factor
    factors = [1] * N
    factors[-1] = factorial_N
    return factors
