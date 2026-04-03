# Large Integer Multiplication

## Problem

Given two large unsigned integers represented as byte arrays in little-endian format, compute their product and return it as a byte array in little-endian format.

## Function Signature

```python
def solve(a_bytes: bytes, b_bytes: bytes) -> bytes:
```

### Parameters
- `a_bytes`: A `bytes` object of length 5,000 representing a large unsigned integer in little-endian byte order.
- `b_bytes`: A `bytes` object of length 5,000 representing a large unsigned integer in little-endian byte order.

### Returns
- A `bytes` object of length 10,000 (len(a_bytes) + len(b_bytes)) representing the product in little-endian byte order.

## Example

```python
>>> # 258 in LE bytes: b'\x02\x01' (2 + 1*256 = 258)
>>> # 3 in LE bytes: b'\x03'
>>> # Product: 774 = b'\x06\x03' in LE (6 + 3*256 = 774)
>>> solve(b'\x02\x01', b'\x03')
b'\x06\x03'
```

## Notes

- Each input integer is 5,000 bytes (~12,000 decimal digits).
- The output must always be exactly `len(a_bytes) + len(b_bytes)` bytes, zero-padded if needed.
- Little-endian means the least significant byte comes first.
- You may use any standard library or numpy for optimization.
