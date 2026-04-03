# Task 12: FizzBuzz

**Source:** https://highload.fun/tasks/12

## Description

You have a data stream of uint32 numbers in binary representation sent to STDIN. The byte order is little endian.

For each number n, print:
- "FizzBuzz" if n is divisible by both 3 and 5
- "Fizz" if n is divisible by 3
- "Buzz" if n is divisible by 5
- n (as a string) if none of the above

## Input

- Binary data via STDIN
- uint32 numbers in little endian byte order
- Numbers quantity: 30,000,000

## Output

- For each number, print the appropriate FizzBuzz result to STDOUT, one per line

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
