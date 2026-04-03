# Matrix Multiplication

## Problem

Given two NxN matrices of floating-point numbers, compute their matrix product.

## Function Signature

```python
def solve(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
```

### Parameters
- `a`: A 350x350 matrix represented as a list of lists of floats. Values are in the range [-10, 10].
- `b`: A 350x350 matrix represented as a list of lists of floats. Values are in the range [-10, 10].

### Returns
- A 350x350 matrix (list of lists of floats) representing the matrix product `a @ b`.

## Example

```python
>>> solve([[1, 2], [3, 4]], [[5, 6], [7, 8]])
[[19, 22], [43, 50]]
```

## Notes

- N = 350 (both matrices are 350x350).
- The result must be correct to within 1e-6 per element compared to the reference answer.
- You may use any standard library or numpy for optimization.
- The naive triple-loop approach is O(N^3) and very slow in pure Python for N=300.
