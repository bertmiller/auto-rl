# Task 5: Median

**Source:** https://highload.fun/tasks/5

## Description

You have a data stream of uint32 numbers in binary representation sent to STDIN. The byte order is little endian.

Try to find the median for the shortest time. The median is the element with the number N/2 (i.e., a[50,000,000] for 100,000,000 elements, using 0-based indexing).

## Input

- Binary data via STDIN
- uint32 numbers in little endian byte order
- Numbers quantity: 100,000,000

## Output

- Print the median value to STDOUT

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
