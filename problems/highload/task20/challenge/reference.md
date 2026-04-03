# Maximum Square

## Problem

Given a binary matrix (containing only 0s and 1s), find the side length of the largest square sub-matrix consisting entirely of 1s. The square must have side length > 1. Return 0 if no such square exists.

## Function Signature

```python
def solve(matrix: list[list[int]]) -> int:
```

### Parameters
- `matrix`: A 5000x5000 binary matrix (list of lists of ints, each 0 or 1). Approximately 50% of entries are 1.

### Returns
- An integer: the side length of the largest all-1s square sub-matrix with side > 1, or 0 if no such square exists.

## Example

```python
>>> solve([
...     [1, 0, 1, 0, 0],
...     [1, 0, 1, 1, 1],
...     [1, 1, 1, 1, 1],
...     [1, 0, 0, 1, 0],
... ])
2
# The largest all-1s square has side length 2 (rows 1-2, cols 2-3)
```

## Notes

- The classic DP approach uses dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 when matrix[i][j] == 1.
- The matrix is 5000x5000, so there are 25,000,000 cells to process.
- You may use any standard library or numpy for optimization.
