"""Verify and score a factorization of 20! into 20 numbers."""

import math


def verify_and_score():
    from solution import solve

    result = solve()

    N = 20
    factorial_N = math.factorial(N)

    # Validate format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    if len(result) != N:
        print("score=-1")
        return

    factors = [int(x) for x in result]

    # Check all positive
    if any(f <= 0 for f in factors):
        print("score=-1")
        return

    # Check product equals N!
    product = 1
    for f in factors:
        product *= f

    if product != factorial_N:
        print("score=-1")
        return

    score = min(factors)
    print(f"score={score}")


if __name__ == "__main__":
    verify_and_score()
