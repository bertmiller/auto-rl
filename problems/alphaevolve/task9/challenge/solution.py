"""Sums and Differences of Finite Sets — maximize log(|A+A|/|A|) / log(|A-A|/|A|)."""


def solve():
    """Return a list of distinct non-negative integers (at least 3).

    Returns:
        list[int]: A set of distinct non-negative integers.
    """
    # Naive solution: arithmetic progression
    # For an AP of length n, |A+A| = 2n-1 and |A-A| = 2n-1
    # So the ratio is log((2n-1)/n) / log((2n-1)/n) = 1.0
    return list(range(50))
