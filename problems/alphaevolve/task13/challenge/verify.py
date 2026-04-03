"""Verify and score an autocorrelation inequality construction."""

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

    heights = np.array(result, dtype=np.float64)

    # Clamp to non-negative
    heights = np.maximum(heights, 0.0)

    # Step function on [-1/4, 1/4] with n intervals
    dx = 0.5 / n  # width of each interval

    # Compute integral of f over [-1/4, 1/4]
    integral_f = np.sum(heights) * dx

    if integral_f < 1e-15:
        print("score=-1e10")
        return

    integral_squared = integral_f ** 2

    # Compute autocorrelation via convolution
    # (f * f)(t) = integral f(t - x) f(x) dx
    # For a step function, the convolution of the height arrays scaled by dx
    # gives the autocorrelation sampled at multiples of dx
    conv = np.convolve(heights, heights) * dx

    # The convolution result has length 2*n - 1
    # conv[k] corresponds to shift t = (k - (n-1)) * dx
    # So t ranges from -(n-1)*dx to (n-1)*dx
    # We need max for |t| <= 1/2, i.e., |k - (n-1)| * dx <= 0.5
    # Since (n-1)*dx = (n-1)*0.5/n < 0.5, all shifts are within |t| <= 0.5
    # So we take max over the entire convolution
    max_autocorr = np.max(conv)

    if max_autocorr < 1e-15:
        print("score=-1e10")
        return

    ratio = max_autocorr / integral_squared
    score = -ratio

    print(f"score={score:.10f}")


if __name__ == "__main__":
    verify_and_score()
