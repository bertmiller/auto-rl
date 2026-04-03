# MD5 Hash

## Problem

Given a byte array, compute its MD5 hash and return it as a lowercase hexadecimal string.

## Function Signature

```python
def solve(data: bytes) -> str:
```

### Parameters
- `data`: A `bytes` object of length 10,000,000 (10 MB) containing arbitrary byte values.

### Returns
- A string: the MD5 hash of the data as a 32-character lowercase hexadecimal string.

## Example

```python
>>> solve(b"hello")
'5d41402abc4b2a76b9719d911017c592'
```

## Notes

- The input is 10 MB of data.
- The output must be exactly 32 lowercase hex characters.
- You may use any standard library or third-party packages for optimization.
- This is already fast with `hashlib.md5()`. The challenge is to find any further optimization, or to recognize when a solution is already near-optimal.
