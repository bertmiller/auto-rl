# Golay Merit Factor (n=63)

## Problem

Find a binary sequence of length 63 (each element +1 or -1) that maximizes the **merit factor**.

## Definitions

For a sequence a[0], a[1], ..., a[n-1] of +/-1 values, the **aperiodic autocorrelation** at lag k is:

    c_k = sum_{j=0}^{n-1-k} a[j] * a[j+k]    for k = 1, 2, ..., n-1

The **energy** of the off-peak autocorrelations is:

    E = 2 * sum_{k=1}^{n-1} c_k^2

The **merit factor** is:

    F = n^2 / E = n^2 / (2 * sum_{k=1}^{n-1} c_k^2)

A higher merit factor means the sequence has smaller off-peak autocorrelations, which is desirable in signal processing, radar, and communications.

## Constraints

- The sequence must have exactly 63 elements.
- Every element must be +1 or -1.

## Objective

Maximize the **merit factor** F.

## Interface

Your `solve()` function must return a `list[int]` of exactly 63 elements, each +1 or -1.

## Scoring

- If the sequence has wrong length or invalid elements, the score is **-1**.
- Otherwise, the score is the merit factor F.

## Notes

- A random +/-1 sequence has expected merit factor ~1.0.
- Barker sequences and Legendre sequences achieve higher merit factors.
- The best known merit factor for length 63 is approximately 14.08 (related to the Legendre sequence for prime 61, extended).
- Merit factor > 5 is already quite good; > 10 is excellent.
- Consider constructions based on quadratic residues modulo a prime.
