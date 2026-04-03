"""Circle Packing in Unit Square (n=15) - Naive grid solution."""


def solve():
    """Return (centers, radii) for 15 circles packed in [0,1]^2.

    Returns:
        tuple: (centers, radii) where centers is a list of (x,y) tuples
               and radii is a list of positive floats.
    """
    n = 15
    # Place circles in a 4x4 grid (using only 15 of 16 spots)
    cols = 4
    rows = 4
    spacing = 1.0 / (2 * cols)
    r = spacing * 0.95  # slightly smaller to avoid touching walls/each other

    centers = []
    radii = []
    count = 0
    for row in range(rows):
        for col in range(cols):
            if count >= n:
                break
            cx = spacing + col * (1.0 / cols)
            cy = spacing + row * (1.0 / rows)
            centers.append((cx, cy))
            radii.append(r)
            count += 1
        if count >= n:
            break

    return centers, radii
