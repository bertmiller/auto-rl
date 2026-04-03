# CRC of Numbers

## Problem

Given binary data containing uint32 little-endian values, compute a custom CRC checksum.

For each uint32 number, convert it to its decimal string representation, then compute a per-number CRC as:
```
sum(digit_value * (position + 1))
```
where position 0 is the leftmost digit.

Return the sum of all per-number CRCs as a uint64 (masked with `& 0xFFFFFFFFFFFFFFFF`).

## Function Signature

```python
def solve(data: bytes) -> int:
```

### Parameters
- `data`: 8,000,000 bytes (2,000,000 uint32 little-endian values).

### Returns
- An integer: the total CRC, masked to 64 bits with `& 0xFFFFFFFFFFFFFFFF`.

## Example

```python
# Number 123: digits "1","2","3"
#   1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
# Number 7: digit "7"
#   7*1 = 7
# Total CRC = 14 + 7 = 21

>>> import struct
>>> data = struct.pack('<II', 123, 7)
>>> solve(data)
21
```

## Notes

- Each uint32 is in the range [0, 4294967295].
- The number 0 has the string "0", so its CRC is 0*1 = 0.
- Position is 1-indexed in the formula (leftmost digit is multiplied by 1).
- There are 2,000,000 numbers to process.
- You may use any standard library or numpy for optimization.
