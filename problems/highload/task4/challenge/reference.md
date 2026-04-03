# Multi-core Unique Strings

## Problem

Given a list of tokens, count the number of unique tokens. This is the same problem as Task 3, but with a larger dataset designed to benefit from multi-core / parallel approaches. The metric is **wall-clock time**, so parallelism helps.

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
- Approximately 4,000,000 unique tokens
- Token charset: `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#%*`
- Maximum token length: 16

## Notes

- This task is scored by **wall-clock time** (not CPU time), so multi-threaded or multi-process approaches can help.
- Consider `multiprocessing`, `concurrent.futures`, or numpy-based approaches.
- The dataset is intentionally large to make parallelism worthwhile.
- You may use any standard library or third-party libraries available in the environment.
