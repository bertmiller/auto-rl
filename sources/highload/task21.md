# Task 21: CRC of Numbers

**Source:** https://highload.fun/tasks/21

## Description

You have a data stream of uint32 numbers in binary representation sent to STDIN. The byte order is little endian.

Try to find the uint64 CRC of these numbers for the shortest time. The CRC formula uses the sum of ASCII digit IDs multiplied by the position of each digit in the decimal representation of each number.

## Input

- Binary data via STDIN
- uint32 numbers in little endian byte order
- Numbers quantity: 250,000,000

## Output

- Print the uint64 CRC value to STDOUT

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
