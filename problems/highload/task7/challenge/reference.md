# Task 7: Prime Number Sum

## Problem

Given binary data containing uint32 little-endian numbers, return the sum of all prime numbers in the data.

## Interface

```python
def solve(data: bytes) -> int:
    """Sum all prime numbers in the binary data.

    Args:
        data: Binary data containing uint32 little-endian numbers.
              Length is always a multiple of 4.

    Returns:
        Sum of all values that are prime numbers (as a Python int, uint64 range).
    """
```

## Constraints

- Input: 50,000 uint32 values packed as little-endian bytes (200,000 bytes total)
- Values range from 0 to 2^32 - 1
- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself
- 0 and 1 are NOT prime

## Primality

Use standard primality testing: a number n > 1 is prime if it has no divisors from 2 to floor(sqrt(n)).

## Example

```python
import struct
data = struct.pack('<3I', 2, 4, 7)  # three uint32s: 2, 4, 7
solve(data)  # returns 9 (2 + 7, since 4 is not prime)
```
