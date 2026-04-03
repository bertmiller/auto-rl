# Task 16: Large Integer Multiplication

**Source:** https://highload.fun/tasks/16

## Description

You have a data stream of exactly 500,000 bytes sent to STDIN. Two unsigned integers are encoded in the stream, each exactly 250,000 bytes in little endian byte order.

Compute the multiplication of the two integers.

## Input

- Binary data via STDIN: exactly 500,000 bytes
- First 250,000 bytes: first unsigned integer (little endian)
- Next 250,000 bytes: second unsigned integer (little endian)

## Output

- Write exactly 500,000 bytes to STDOUT containing the multiplication result in little endian byte order

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
