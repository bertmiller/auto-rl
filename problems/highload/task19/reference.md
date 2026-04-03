# Blue Channel Extraction from RGB

## Problem

Given raw RGB pixel data as a byte array (3 bytes per pixel: R, G, B), extract and return only the blue channel bytes.

## Function Signature

```python
def solve(data: bytes) -> bytes:
```

### Parameters
- `data`: A `bytes` object of length 150,000,000 (50,000,000 pixels x 3 bytes per pixel). Every group of 3 consecutive bytes represents one pixel in R, G, B order.

### Returns
- A `bytes` object of length 3,000,000 containing only the blue channel byte from each pixel (i.e., every 3rd byte starting at index 2).

## Example

```python
>>> solve(b'\xff\x00\x80\x10\x20\x30')
b'\x80\x30'
# Pixel 0: R=0xff, G=0x00, B=0x80 -> blue = 0x80
# Pixel 1: R=0x10, G=0x20, B=0x30 -> blue = 0x30
```

## Notes

- The input has exactly 50,000,000 pixels (150,000,000 bytes).
- The output has exactly 50,000,000 bytes (one blue byte per pixel).
- You may use any standard library or numpy for optimization.
