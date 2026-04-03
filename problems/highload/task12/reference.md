# FizzBuzz

## Problem

Given binary data containing uint32 little-endian integers, return the FizzBuzz output as a single string with one result per line.

For each number `n`:
- If `n % 15 == 0`: output `"FizzBuzz"`
- Else if `n % 3 == 0`: output `"Fizz"`
- Else if `n % 5 == 0`: output `"Buzz"`
- Else: output `str(n)`

Results are joined by newlines (with a trailing newline).

## Function Signature

```python
def solve(data: bytes) -> str:
```

### Parameters
- `data`: Raw bytes containing 20,000,000 uint32 values in little-endian format (4 bytes each, 80MB total).

### Returns
- A string: the FizzBuzz output, one result per line, with a trailing newline.

## Example

```python
>>> import struct
>>> data = struct.pack('<3I', 1, 3, 15)
>>> solve(data)
'1\nFizz\nFizzBuzz\n'
```

## Notes

- Input contains exactly 20,000,000 uint32 values (80,000,000 bytes).
- Numbers range from 0 to 2^32 - 1.
- Zero is divisible by everything, so `solve(struct.pack('<I', 0))` returns `"FizzBuzz\n"`.
- You may use any standard library or numpy for optimization.
