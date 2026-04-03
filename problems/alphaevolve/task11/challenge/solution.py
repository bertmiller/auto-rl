"""Golay Merit Factor (n=63) — maximize the merit factor of a binary sequence."""


def solve():
    """Return a list of exactly 63 values, each +1 or -1.

    Returns:
        list[int]: A binary (+/-1) sequence of length 63.
    """
    # Naive solution: pseudo-random sequence using middle bits of LCG.
    # Expected merit factor ~ 1.0 for a random-looking sequence.
    n = 63
    seq = []
    state = 7
    for _ in range(n):
        state = (state * 48271) % 2147483647  # Park-Miller LCG
        seq.append(1 if (state >> 16) % 2 == 0 else -1)
    return seq
