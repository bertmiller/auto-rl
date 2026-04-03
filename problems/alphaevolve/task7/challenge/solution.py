"""Squares in a Square (n=10) - Naive grid solution."""


def solve():
    """Return 10 axis-aligned squares packed inside [0,1]^2.

    Returns:
        list: A list of 10 (x, y, side_length) tuples where (x, y)
              is the bottom-left corner.
    """
    n = 10
    # Place 10 squares in a 2x5 grid
    cols = 5
    rows = 2
    side = 1.0 / cols  # 0.2

    squares = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            x = col * side
            y = row * (1.0 / rows)
            squares.append((x, y, side))
            count += 1
        if count >= n:
            break

    return squares
