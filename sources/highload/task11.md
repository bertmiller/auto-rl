# Task 11: RFC3339 Timestamps

**Source:** https://highload.fun/tasks/11

## Description

You have a data stream of datetimes in RFC3339 format sent to STDIN. Each line contains one datetime.

Try to find the sum (int64) of UTC timestamps for the shortest time.

## Input

- RFC3339 datetime strings via STDIN, one per line
- Datetimes are between 1950-01-01T00:00:00 and 2050-12-31T23:59:59
- Number of lines: 5,000,000

## Output

- Print the int64 sum of all UTC timestamps (Unix epoch seconds) to STDOUT

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
