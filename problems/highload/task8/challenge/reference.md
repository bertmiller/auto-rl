# Task 8: Top 100 Greatest Numbers

## Problem

Given binary data containing uint32 little-endian numbers, return the sum of the top 100 largest values.

## Interface

```python
def solve(data: bytes) -> int:
    """Return the sum of the top 100 largest uint32 values.

    Args:
        data: Binary data containing uint32 little-endian numbers.
              Length is always a multiple of 4.

    Returns:
        Sum of the 100 largest values (as a Python int).
    """
```

## Constraints

- Input: 5,000,000 uint32 values packed as little-endian bytes (20,000,000 bytes total)
- Values range from 0 to 2^32 - 1
- Return the sum of the 100 largest values
- If there are fewer than 100 values, return the sum of all values

## Example

```python
import struct
data = struct.pack('<5I', 10, 50, 30, 40, 20)
# Top 100 (or all 5): 50 + 40 + 30 + 20 + 10 = 150
solve(data)  # returns 150
```
