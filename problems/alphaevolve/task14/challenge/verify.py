"""Verify and score an Erdos minimum overlap construction."""

import numpy as np


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate format
    if not isinstance(result, (list, tuple)):
        print("score=-1e10")
        return

    n = len(result)
    if n < 2:
        print("score=-1e10")
        return

    h = np.array(result, dtype=np.float64)

    # Clamp to [0, 1]
    h = np.clip(h, 0.0, 1.0)

    dx = 2.0 / n  # width of each interval on [0, 2]

    # Normalize so integral = 1 (mean = 0.5)
    current_mean = np.mean(h)
    if current_mean < 1e-15:
        print("score=-1e10")
        return
    h = h * (0.5 / current_mean)
    # Re-clamp after normalization
    h = np.clip(h, 0.0, 1.0)

    # Compute max over shifts k of integral h(x)(1 - h(x+k)) dx
    # For each shift k (in units of intervals), compute the overlap
    # h is defined on [0, 2] with n intervals
    # h(x+k) shifts h to the left by k intervals
    # The overlap integral is over the region where both h(x) and h(x+k) are defined

    max_overlap = 0.0

    # Consider shifts from -(n-1) to (n-1) intervals
    for shift in range(-(n - 1), n):
        # Compute integral of h(x) * (1 - h(x + shift*dx)) dx
        # over the region where both are defined
        if shift >= 0:
            h_x = h[:n - shift]
            h_shifted = h[shift:]
        else:
            h_x = h[-shift:]
            h_shifted = h[:n + shift]

        overlap = np.sum(h_x * (1.0 - h_shifted)) * dx
        if overlap > max_overlap:
            max_overlap = overlap

    score = -max_overlap
    print(f"score={score:.10f}")


if __name__ == "__main__":
    verify_and_score()
