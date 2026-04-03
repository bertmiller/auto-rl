"""Verify and score an Erdos discrepancy sequence (D=3)."""

import sys

MAX_LEN = 5000
D = 3


def compute_discrepancy(x):
    """Compute the discrepancy of a +/-1 sequence.

    disc(x) = max over d>=1, n>=1 of |sum_{k=1}^{n} x[k*d]|
    Uses 1-indexed logic: x[k*d] means the element at position k*d.
    """
    N = len(x)
    max_disc = 0
    for d in range(1, N + 1):
        partial_sum = 0
        k = d  # first element in this progression
        while k <= N:
            partial_sum += x[k - 1]  # convert to 0-indexed
            abs_val = abs(partial_sum)
            if abs_val > max_disc:
                max_disc = abs_val
                if max_disc > D:
                    return max_disc  # early exit
            k += d
    return max_disc


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    # Cap length
    if len(result) > MAX_LEN:
        result = result[:MAX_LEN]

    if len(result) == 0:
        print("score=-1")
        return

    # Check all elements are +1 or -1
    for i, x in enumerate(result):
        if x not in (1, -1):
            print("score=-1")
            return

    # Compute discrepancy
    disc = compute_discrepancy(result)

    if disc > D:
        print("score=-1")
        return

    score = len(result)
    print(f"score={score}")


if __name__ == "__main__":
    verify_and_score()
