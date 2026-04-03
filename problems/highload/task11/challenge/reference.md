# RFC3339 Timestamps

## Problem

Given a list of RFC3339 datetime strings with timezone offsets, return the sum of their Unix timestamps (seconds since 1970-01-01T00:00:00 UTC) as a plain integer.

## Function Signature

```python
def solve(timestamps: list[str]) -> int:
```

### Parameters
- `timestamps`: A list of 2,000,000 RFC3339 datetime strings. Each string has the format `YYYY-MM-DDTHH:MM:SS+HH:MM` or `YYYY-MM-DDTHH:MM:SS-HH:MM`. Datetimes range from 1950-01-01T00:00:00 to 2050-12-31T23:59:59 (before 1970 yields negative Unix timestamps).

### Returns
- An integer: the sum of all Unix timestamps (epoch seconds). The sum is a plain Python integer (may be negative for individual timestamps before 1970, but the total sum will be positive given the data distribution).

## Example

```python
>>> solve(["2023-05-15T10:30:00+03:00", "1960-01-01T00:00:00+00:00"])
# 2023-05-15T10:30:00+03:00 -> UTC 07:30:00 -> 1684135800
# 1960-01-01T00:00:00+00:00 -> -315619200
# returns 1684135800 + (-315619200) = 1368516600
```

## Notes

- Timestamps before 1970-01-01 produce negative Unix timestamps. This is expected behavior.
- Timezone offsets range from -12:00 to +14:00.
- You must correctly handle timezone conversion to UTC before computing the epoch second.
- You may use any standard library or third-party library (numpy, etc.) for optimization.
