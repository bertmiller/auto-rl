"""
Circle Packing in a Square — Variable Radii

Pack N non-overlapping circles with variable radii into the unit square
[0,1]^2, maximizing the **sum of radii**.

This is the canonical problem studied by AlphaEvolve (DeepMind, 2025) and
ShinkaEvolve (Sakana AI, 2025).  Reference database: Erich Friedman's
Packing Center, with previous bests mostly by David W. Cantrell (2011).

Architecture (mirrors AlphaEvolve):
  The LLM evolves `construct_packing()` — a search algorithm that, given n
  and a time budget, returns (centers, radii).  Everything outside the
  EVOLVE-BLOCK markers is fixed infrastructure.
"""

import math
import time

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Evaluation (fixed infrastructure)
# ---------------------------------------------------------------------------

def validate(centers: np.ndarray, radii: np.ndarray, tol: float = 1e-9) -> bool:
    """Check that a packing is feasible."""
    n = len(centers)
    if n == 0 or len(radii) != n:
        return False

    # Containment: each circle inside [0,1]^2
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False

    # No overlaps: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False

    return True


def score(centers: np.ndarray, radii: np.ndarray) -> float:
    """Return sum of radii if valid, else 0."""
    if not validate(centers, radii):
        return 0.0
    return float(np.sum(radii))


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START — the LLM mutates everything between these markers
# ---------------------------------------------------------------------------

def construct_packing(n: int, time_budget: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Search for a good packing of n circles in [0,1]^2.

    This is the evolvable function.  AlphaEvolve / ShinkaEvolve would
    mutate this entire function body to discover better search strategies.

    Args:
        n:           Number of circles.
        time_budget: Wall-clock seconds allowed.

    Returns:
        (centers, radii) — both shape (n,2) and (n,) respectively.
    """
    deadline = time.monotonic() + time_budget
    best_centers = None
    best_radii = None
    best_score = -1.0

    # --- Greedy construction --------------------------------------------------

    def greedy_construct(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Place circles one at a time, largest-first, in the biggest gap."""
        centers = np.empty((n, 2))
        radii = np.empty(n)

        placed = 0
        for i in range(n):
            # Sample candidate positions, pick the one with most space
            n_cands = 2000
            cands = rng.random((n_cands, 2))

            # Max radius at each candidate = min distance to wall or placed circle
            wall_clearance = np.minimum(
                np.minimum(cands[:, 0], 1.0 - cands[:, 0]),
                np.minimum(cands[:, 1], 1.0 - cands[:, 1]),
            )

            if placed == 0:
                max_r = wall_clearance
            else:
                # Distance to each placed circle minus its radius
                diffs = cands[:, np.newaxis, :] - centers[np.newaxis, :placed, :]  # (M,K,2)
                dists = np.sqrt((diffs ** 2).sum(axis=2))  # (M, K)
                clearance_from_circles = (dists - radii[:placed]).min(axis=1)
                max_r = np.minimum(wall_clearance, clearance_from_circles)

            max_r = np.maximum(max_r, 0.0)
            best_idx = np.argmax(max_r)

            centers[i] = cands[best_idx]
            radii[i] = max_r[best_idx]
            placed += 1

        return centers, radii

    # --- SLSQP refinement ----------------------------------------------------

    def refine_slsqp(centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint optimisation of centers and radii via SLSQP."""
        nc = len(centers)

        # Pack into flat vector: [x0,y0,r0, x1,y1,r1, ...]
        x0 = np.empty(3 * nc)
        for i in range(nc):
            x0[3 * i] = centers[i, 0]
            x0[3 * i + 1] = centers[i, 1]
            x0[3 * i + 2] = radii[i]

        def neg_sum_radii(x):
            return -sum(x[3 * i + 2] for i in range(nc))

        constraints = []

        # Containment: x_i - r_i >= 0, 1 - x_i - r_i >= 0 (same for y)
        for i in range(nc):
            xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
            constraints.append({"type": "ineq", "fun": lambda x, _xi=xi, _ri=ri: x[_xi] - x[_ri]})
            constraints.append({"type": "ineq", "fun": lambda x, _xi=xi, _ri=ri: 1.0 - x[_xi] - x[_ri]})
            constraints.append({"type": "ineq", "fun": lambda x, _yi=yi, _ri=ri: x[_yi] - x[_ri]})
            constraints.append({"type": "ineq", "fun": lambda x, _yi=yi, _ri=ri: 1.0 - x[_yi] - x[_ri]})

        # Non-overlap: dist(i,j) - r_i - r_j >= 0
        for i in range(nc):
            for j in range(i + 1, nc):
                xi, yi, ri = 3 * i, 3 * i + 1, 3 * i + 2
                xj, yj, rj = 3 * j, 3 * j + 1, 3 * j + 2

                def overlap_constraint(x, _xi=xi, _yi=yi, _ri=ri, _xj=xj, _yj=yj, _rj=rj):
                    dx = x[_xi] - x[_xj]
                    dy = x[_yi] - x[_yj]
                    return math.sqrt(dx * dx + dy * dy) - x[_ri] - x[_rj]

                constraints.append({"type": "ineq", "fun": overlap_constraint})

        bounds = []
        for i in range(nc):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

        result = minimize(
            neg_sum_radii,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        c_out = np.empty((nc, 2))
        r_out = np.empty(nc)
        for i in range(nc):
            c_out[i, 0] = result.x[3 * i]
            c_out[i, 1] = result.x[3 * i + 1]
            r_out[i] = result.x[3 * i + 2]

        return c_out, r_out

    # --- Multi-restart search loop -------------------------------------------

    seed = 0
    while time.monotonic() < deadline:
        rng = np.random.default_rng(seed)
        seed += 1

        # Construct
        centers, radii = greedy_construct(rng)

        # Refine
        try:
            centers, radii = refine_slsqp(centers, radii)
        except Exception:
            pass

        # Evaluate
        if validate(centers, radii):
            s = float(np.sum(radii))
            if s > best_score:
                best_score = s
                best_centers = centers.copy()
                best_radii = radii.copy()

    if best_centers is None:
        # Fallback: single circle at centre
        best_centers = np.full((n, 2), 0.5)
        best_radii = np.full(n, 0.5 / n)

    return best_centers, best_radii


# EVOLVE-BLOCK-END

# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def solve(n: int, time_budget: float = 30.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the circle packing problem.

    Returns:
        (centers, radii, sum_of_radii)
    """
    centers, radii = construct_packing(n, time_budget=time_budget)
    return centers, radii, score(centers, radii)


if __name__ == "__main__":
    import sys

    ns = range(1, 21) if len(sys.argv) < 2 else [int(sys.argv[1])]
    for n in ns:
        _, _, s = solve(n, time_budget=10.0)
        print(f"N={n:3d}  sum_radii={s:.6f}")
