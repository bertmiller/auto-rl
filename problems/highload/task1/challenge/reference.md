# Sum of Numbers (Parse Integers)

## Problem

Given a list of number strings, each representing an integer in the range [0, 2^31 - 1], return their sum masked to a 64-bit unsigned integer (i.e., result & 0xFFFFFFFFFFFFFFFF).

## Function Signature

```python
def solve(numbers: list[str]) -> int:
```

### Parameters
- `numbers`: A list of 30,000,000 strings, each representing a non-negative integer in [0, 2^31 - 1].

### Returns
- An integer: the sum of all parsed numbers, masked with `& 0xFFFFFFFFFFFFFFFF`.

## Example

```python
>>> solve(["1", "2", "3", "4294967295"])
# returns (1 + 2 + 3 + 4294967295) & 0xFFFFFFFFFFFFFFFF = 4294967301
```

## Notes

- The input list contains 30,000,000 elements.
- Each string is a valid non-negative integer that fits in a 32-bit unsigned range.
- The mask `0xFFFFFFFFFFFFFFFF` ensures the result fits in 64 bits. In practice, Python handles arbitrary precision, so this mainly ensures consistent output.
- You may use any standard library or numpy for optimization.
