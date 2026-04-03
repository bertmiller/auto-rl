# Blue from RGBA

## Problem

Given raw RGBA pixel data (4 bytes per pixel: R, G, B, A), extract and return only the blue channel bytes.

## Function Signature

```python
def solve(data: bytes) -> bytes:
```

### Parameters
- `data`: Raw bytes containing 60,000,000 RGBA pixels. Each pixel is 4 bytes in order: Red, Green, Blue, Alpha. Total size: 240,000,000 bytes.

### Returns
- A bytes object containing only the blue channel values, one byte per pixel, in the same pixel order. Length: 60,000,000 bytes.

## Example

```python
>>> # Two pixels: (R=10, G=20, B=30, A=40) and (R=50, G=60, B=70, A=80)
>>> solve(bytes([10, 20, 30, 40, 50, 60, 70, 80]))
b'\x1e\x46'  # bytes([30, 70])
```

## Notes

- Input length is always a multiple of 4.
- The blue channel is at offset 2 within each 4-byte pixel group (indices 2, 6, 10, ...).
- Input is 240,000,000 bytes (240MB), output is 60,000,000 bytes (60MB).
- You may use any standard library or numpy for optimization.
