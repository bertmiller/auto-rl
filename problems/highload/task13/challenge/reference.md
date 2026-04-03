# Count uint8

## Problem

Given a bytes object, count how many bytes have the value 127.

## Function Signature

```python
def solve(data: bytes) -> int:
```

### Parameters
- `data`: A bytes object of length 200,000,000 containing random byte values (0-255).

### Returns
- An integer: the number of bytes in `data` that are equal to 127.

## Example

```python
>>> solve(b'\x7f\x00\x7f\x01')
2
```

## Notes

- The input is 200,000,000 bytes (approximately 200MB).
- Byte values are uniformly distributed, so roughly 200,000,000 / 256 ~ 781,250 bytes will equal 127.
- You may use any standard library or numpy for optimization.
