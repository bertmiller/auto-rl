# Unique Strings

## Problem

Given a list of tokens, count the number of unique tokens.

## Function Signature

```python
def solve(tokens: list[str]) -> int:
```

### Parameters
- `tokens`: A list of 20,000,000 strings. Each token is composed of characters from the set: `0-9`, `a-z`, `A-Z`, `@`, `#`, `%`, `*`. Maximum token length is 16.

### Returns
- An integer: the count of unique (distinct) tokens in the list.

## Example

```python
>>> solve(["hello", "world", "hello", "foo"])
3
```

## Data Characteristics

- 20,000,000 total tokens
- Approximately 2,000,000 unique tokens
- Token charset: `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#%*`
- Maximum token length: 16

## Notes

- You may use any standard library or numpy for optimization.
- The straightforward approach is `len(set(tokens))`, but there may be faster approaches.
