"""Block Stacking Overhang (n=20) - Classic harmonic stacking solution."""


def solve():
    """Return 20 x-positions (left edges) for stacking blocks.

    The table edge is at x=0 (table extends to the left).
    Block i occupies [x[i], x[i]+1]. Block 0 is on the table, block 19 on top.

    Returns:
        list: A list of 20 floats (x-positions of left edges).
    """
    n = 20

    # Classic harmonic stacking: build offsets between consecutive blocks
    # offset between block i and block i+1 = 1 / (2 * (n - 1 - i))
    # (from top: top block overhangs by 1/2, next pair by 1/4, etc.)
    offsets = [1.0 / (2.0 * (n - 1 - i)) for i in range(n - 1)]

    # Compute positions starting from x[0] = 0
    x = [0.0] * n
    for i in range(1, n):
        x[i] = x[i - 1] + offsets[i - 1]

    # Compute total center of mass
    total_com = sum(xj + 0.5 for xj in x) / n

    # Shift all blocks left so that total COM = 0 (at table edge)
    for i in range(n):
        x[i] -= total_com

    return x
