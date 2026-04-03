# Median

## Problem

Given binary data containing uint32 little-endian values, find the median element. The median is defined as the element at index `N // 2` when the values are sorted in ascending order (0-indexed).

## Function Signature

```python
def solve(data: bytes) -> int:
```

### Parameters
- `data`: A bytes object containing 10,000,000 uint32 values in little-endian format (total 40,000,000 bytes).

### Returns
- An integer: the value at index `N // 2` (i.e., index 5,000,000) when all 10,000,000 values are sorted in ascending order.

## Example

```python
import struct
values = [5, 3, 1, 4, 2]
data = struct.pack(f"<{len(values)}I", *values)
# sorted: [1, 2, 3, 4, 5], N=5, index N//2 = 2
# solve(data) returns 3
```

## Data Size

- 10,000,000 uint32 values (40 MB of binary data)
- Values are random in [0, 2^32 - 1]

## Notes

- The data is always in little-endian uint32 format.
- You may use `struct`, `array`, `numpy`, or any other approach to parse and process the data.
- Consider that you don't necessarily need a full sort to find the median -- selection algorithms (like `numpy.partition`) can be faster.
