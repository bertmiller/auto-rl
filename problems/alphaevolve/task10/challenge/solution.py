"""Erdos Discrepancy (D=3) — find the longest +/-1 sequence with discrepancy <= 3."""


def solve():
    """Return a list of +1/-1 values with discrepancy at most 3.

    Returns:
        list[int]: A sequence of +1 and -1 values.
    """
    # Naive solution: greedy-constructed sequence of length 109 with discrepancy <= 3.
    # Found by greedy search with random seed 144.
    # The challenge is to find much longer sequences (up to 5000).
    return [
        1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1,
        1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1,
        1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1,
        1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1,
        1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1,
        -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1,
        1,
    ]
