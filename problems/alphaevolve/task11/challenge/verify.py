"""Verify and score a Golay merit factor construction (n=63)."""

import sys

N = 63


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate return format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    if len(result) != N:
        print("score=-1")
        return

    # Check all elements are +1 or -1
    for x in result:
        if x not in (1, -1):
            print("score=-1")
            return

    # Compute aperiodic autocorrelations
    a = list(result)
    energy = 0.0
    for k in range(1, N):
        c_k = 0
        for j in range(N - k):
            c_k += a[j] * a[j + k]
        energy += c_k * c_k

    energy *= 2  # E = 2 * sum of c_k^2

    if energy == 0:
        # Perfect sequence (impossible for n>1, but handle edge case)
        print(f"score={float('inf')}")
        return

    merit_factor = (N * N) / energy
    print(f"score={merit_factor:.6f}")


if __name__ == "__main__":
    verify_and_score()
