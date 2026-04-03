"""No Isosceles Triangles on Grid (n=15) — maximize points with no isosceles triple."""


def solve():
    """Return a list of (x, y) grid points in [0,14]^2 with no isosceles triple.

    Returns:
        list[tuple[int, int]]: List of distinct grid points.
    """
    # Greedy construction: add points one by one, checking the constraint.
    import random
    random.seed(42)

    def sq_dist(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    def is_valid(points, new_point):
        """Check if adding new_point creates any isosceles triple."""
        n = len(points)
        # Check all pairs from existing points combined with new_point
        for i in range(n):
            d_new_i = sq_dist(new_point, points[i])
            for j in range(i + 1, n):
                d_new_j = sq_dist(new_point, points[j])
                d_i_j = sq_dist(points[i], points[j])
                # All three distances must be distinct
                if d_new_i == d_new_j or d_new_i == d_i_j or d_new_j == d_i_j:
                    return False
        return True

    # Generate all grid points and shuffle
    all_points = [(x, y) for x in range(15) for y in range(15)]
    random.shuffle(all_points)

    selected = []
    for p in all_points:
        if is_valid(selected, p):
            selected.append(p)

    return selected
