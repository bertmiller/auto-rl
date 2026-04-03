# Task 6: Sort UUID

## Problem

Given a list of UUID strings, sort them in ascending lexicographic order.

## Interface

```python
def solve(uuids: list[str]) -> list[str]:
    """Sort UUID strings in ascending lexicographic order.

    Args:
        uuids: List of UUID strings (e.g., "550e8400-e29b-41d4-a716-446655440000")

    Returns:
        List of UUID strings sorted in ascending lexicographic order.
    """
```

## Constraints

- Input: 5,000,000 UUID strings in standard format (8-4-4-4-12 hex digits)
- Output: Same UUIDs sorted lexicographically (ascending)
- All UUIDs are valid and unique

## Example

```python
>>> solve(["c0a80001-0000-0000-0000-000000000000", "a0a80001-0000-0000-0000-000000000000"])
["a0a80001-0000-0000-0000-000000000000", "c0a80001-0000-0000-0000-000000000000"]
```
