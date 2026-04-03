"""Verify and score a sumset/difference-set construction."""

import math
import sys


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    # Check all elements are integers
    for x in result:
        if not isinstance(x, (int,)):
            # Allow numpy ints
            try:
                if int(x) != x:
                    print("score=-1")
                    return
            except (TypeError, ValueError):
                print("score=-1")
                return

    A = list(set(int(x) for x in result))

    # Check non-negative
    if any(x < 0 for x in A):
        print("score=-1")
        return

    # Check at least 3 distinct elements
    if len(A) < 3:
        print("score=-1")
        return

    n = len(A)

    # Compute sumset A+A
    sumset = set()
    for a in A:
        for b in A:
            sumset.add(a + b)

    # Compute difference set A-A
    diffset = set()
    for a in A:
        for b in A:
            diffset.add(a - b)

    ss = len(sumset)
    ds = len(diffset)

    ratio_sum = ss / n
    ratio_diff = ds / n

    # Both ratios must be > 1
    if ratio_diff <= 1.0 or ratio_sum <= 1.0:
        print("score=-1")
        return

    score = math.log(ratio_sum) / math.log(ratio_diff)
    print(f"score={score:.6f}")


if __name__ == "__main__":
    verify_and_score()
