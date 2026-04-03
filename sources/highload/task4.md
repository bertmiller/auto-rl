# Task 4: Multi-core Unique Strings

**Source:** https://highload.fun/tasks/4

## Description

A task similar to "Unique Strings" but with 3 CPU kernels available. The score function depends on wall time instead of CPU time.

You have a data stream of string tokens sent to STDIN. Each line of the flow has one token.

Try to find the exact number of unique tokens for the shortest time.

## Input

- String tokens via STDIN, one per line
- Each token is composed of characters from the set: `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#%*`
- Maximum token length: up to 16 characters
- Number of unique tokens: approximately 1,000,000

## Output

- Print the exact count of unique tokens to STDOUT

## Constraints

- 3 CPU kernels available
- Score depends on **wall time** (not CPU time)
- Languages: C, C++, Go, Rust
