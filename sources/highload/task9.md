# Task 9: Arithmetic Expression Evaluation

**Source:** https://highload.fun/tasks/9

## Description

You have a data stream of arithmetic expressions sent to STDIN. Each line of the flow has one expression.

Evaluate each expression and print the result (int64) to STDOUT for the shortest time.

## Input

- Arithmetic expressions via STDIN, one per line
- Each expression contains approximately 50,000 int16 numbers
- Basic operations: `+`, `-`, `*`, `/` (integer division)
- Parentheses `(` and `)` are supported

## Output

- For each expression, print the int64 result to STDOUT

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
