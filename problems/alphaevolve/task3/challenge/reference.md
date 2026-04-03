# Thomson Problem (n=30)

## Problem

Place 30 points on the surface of the unit sphere to minimize the Coulomb energy (electrostatic potential energy).

The Coulomb energy is defined as:

    E = sum_{i < j} 1 / dist(i, j)

where dist(i, j) is the Euclidean distance between points i and j.

## Constraints

- You must return exactly 30 points.
- Each point (x, y, z) will be normalized to the unit sphere by the verifier.
- Points must be finite (no NaN or Inf).
- No two points may be coincident (distance must be > 0).

## Objective

Maximize the **negative Coulomb energy**: score = -E.

Since E is always positive, the score is always negative. A higher (less negative) score means lower energy, which is better.

## Interface

Your `solve()` function must return a list of 30 `(x, y, z)` tuples.

## Scoring

- If fewer than 30 points, invalid values, or coincident points: score = **-1e10**.
- Otherwise: score = -sum_{i<j} 1/dist(i,j).

## Notes

- A Fibonacci sphere gives a reasonable starting point with energy around 400-420 (score around -420).
- The theoretical minimum energy for 30 points is approximately 388.79 (score ~ -388.79).
- Consider gradient descent on the energy function, simulated annealing, or repulsion dynamics.
- Small perturbations and local optimization can significantly improve over the naive solution.
