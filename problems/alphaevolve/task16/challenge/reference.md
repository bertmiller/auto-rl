# Factoring N! into N Numbers (N=20)

## Problem

Given N = 20, find exactly 20 positive integers whose product equals 20! = 2432902008176640000.

The goal is to **maximize the minimum** of the 20 factors. That is, make the factorization as balanced as possible.

## Key Facts

- 20! = 2432902008176640000
- 20!^(1/20) is approximately 48893, so a perfectly balanced factorization would have all factors around this value.
- Of course, 20! has specific prime factors, so a perfect split is unlikely, but getting close is the challenge.

## Interface

`solve()` returns a `list[int]` of exactly 20 positive integers whose product equals 20!.

## Scoring

The verifier:
1. Checks that exactly 20 positive integers are returned
2. Checks that their product equals 20! (using Python's exact integer arithmetic)
3. Returns min(factors) as the score

- If the product does not equal 20!, returns score = -1.
- If any factor is <= 0, returns score = -1.
- If the count is not exactly 20, returns score = -1.

## Notes

- Naive: [1, 1, ..., 1, 20!] gives min = 1.
- Better: try to split large prime factors and balance the product.
- The prime factorization of 20! is: 2^18 * 3^8 * 5^4 * 7^2 * 11 * 13 * 17 * 19.
- Use Python's arbitrary precision integers. Compute 20! as math.factorial(20).
- Consider greedy approaches: repeatedly split the largest factor, or assign prime factors to the smallest current factor.
