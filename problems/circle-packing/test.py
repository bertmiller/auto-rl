"""
Evaluation for circle packing solutions.

Validates packings and compares achieved sum-of-radii against known best values.
Run with ``pytest test.py`` or ``python test.py`` for a full report.

Problem: Pack N non-overlapping circles with variable radii into [0,1]^2,
maximizing the sum of radii.

Reference: Erich Friedman's Packing Center
  https://erich-friedman.github.io/packing/cirRsqu/
"""

import math

import numpy as np

from solution import score, solve, validate

# ---------------------------------------------------------------------------
# Best known sum-of-radii for N circles in [0,1]^2  (variable radii)
#
# Source: Erich Friedman's Packing Center.  Previous bests mostly by
# David W. Cantrell (2011); N=26,32 improved by AlphaEvolve (2025).
# ---------------------------------------------------------------------------

KNOWN_BEST: dict[int, float] = {
    1:  0.500,
    2:  0.585,
    3:  0.796,
    4:  1.006,
    5:  1.103,
    6:  1.202,
    7:  1.306,
    8:  1.423,
    9:  1.524,
    10: 1.591,
    11: 1.680,
    12: 1.765,
    13: 1.829,
    14: 1.905,
    15: 1.980,
    16: 2.053,
    17: 2.111,
    18: 2.178,
    19: 2.236,
    20: 2.301,
    21: 2.362,
    22: 2.420,
    23: 2.478,
    24: 2.530,
    25: 2.587,
    26: 2.636,   # AlphaEvolve improved to ~2.63586
    27: 2.685,
    28: 2.737,
    29: 2.790,
    30: 2.842,
    31: 2.889,
    32: 2.939,   # AlphaEvolve improved to ~2.93794
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def violation_report(centers: np.ndarray, radii: np.ndarray, tol: float = 1e-9) -> list[str]:
    """Detailed feasibility check.  Returns list of violation messages."""
    errors: list[str] = []
    n = len(centers)
    if n == 0:
        return errors

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < -tol:
            errors.append(f"Circle {i}: negative radius {r:.8f}")
        if x - r < -tol:
            errors.append(f"Circle {i}: left boundary violation x-r={x - r:.8f}")
        if x + r > 1 + tol:
            errors.append(f"Circle {i}: right boundary violation x+r={x + r:.8f}")
        if y - r < -tol:
            errors.append(f"Circle {i}: bottom boundary violation y-r={y - r:.8f}")
        if y + r > 1 + tol:
            errors.append(f"Circle {i}: top boundary violation y+r={y + r:.8f}")

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            gap = dist - radii[i] - radii[j]
            if gap < -tol:
                errors.append(
                    f"Circles {i},{j}: overlap by {-gap:.8f} "
                    f"(dist={dist:.6f}, r_i+r_j={radii[i] + radii[j]:.6f})"
                )

    return errors


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------

def test_packing_valid():
    """All packings (N=1..10) should be geometrically valid."""
    for n in range(1, 11):
        centers, radii, s = solve(n, time_budget=5.0)
        assert validate(centers, radii), f"N={n}: invalid packing"


def test_scores_positive():
    """All packings should produce positive sum-of-radii."""
    for n in range(1, 11):
        _, _, s = solve(n, time_budget=5.0)
        assert s > 0, f"N={n}: score={s}"


def test_known_bests():
    """Packings should achieve >= 90% of known best sum-of-radii."""
    for n in [1, 2, 3, 4, 5, 9, 10]:
        _, _, s = solve(n, time_budget=10.0)
        best = KNOWN_BEST[n]
        ratio = s / best
        assert ratio >= 0.90, (
            f"N={n}: sum_radii={s:.6f}, best_known={best:.6f}, ratio={ratio:.4f}"
        )


def test_sum_radii_increasing():
    """Sum of radii should generally increase with N."""
    scores = []
    for n in range(1, 8):
        _, _, s = solve(n, time_budget=5.0)
        scores.append(s)
    assert scores[0] < scores[-1], (
        f"s(1)={scores[0]:.6f} should be < s(7)={scores[-1]:.6f}"
    )


def test_n1_exact():
    """N=1 should be trivially optimal: single circle with r=0.5."""
    _, _, s = solve(1, time_budget=2.0)
    assert abs(s - 0.5) < 1e-3, f"N=1: expected sum~0.5, got {s:.6f}"


def test_n2_diagonal():
    """N=2 should be close to optimal (two circles on diagonal)."""
    _, _, s = solve(2, time_budget=10.0)
    expected = 2 - math.sqrt(2)  # ~0.5858
    ratio = s / expected
    assert ratio >= 0.95, f"N=2: sum_radii={s:.6f}, expected={expected:.6f}"


# ---------------------------------------------------------------------------
# Standalone report
# ---------------------------------------------------------------------------

def report(max_n: int = 20, time_budget: float = 15.0) -> None:
    """Print a full evaluation table."""
    print("Circle Packing Evaluation  (variable radii, maximize sum)")
    print("=" * 70)
    print(f"{'N':>4}  {'Sum Radii':>10}  {'Best Known':>10}  {'Ratio':>8}  {'Valid':>6}")
    print("-" * 70)

    for n in range(1, max_n + 1):
        centers, radii, s = solve(n, time_budget=time_budget)
        errors = violation_report(centers, radii)
        best = KNOWN_BEST.get(n)

        best_str = f"{best:.6f}" if best else "       --"
        ratio_str = f"{s / best:.4f}" if best else "   --"
        valid_str = "OK" if not errors else "FAIL"

        print(f"{n:4d}  {s:10.6f}  {best_str:>10}  {ratio_str:>8}  {valid_str:>6}")
        for err in errors[:3]:
            print(f"        ! {err}")

    print("=" * 70)


if __name__ == "__main__":
    report()
