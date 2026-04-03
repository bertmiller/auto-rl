# Task 19: Blue from RGB

**Source:** https://highload.fun/tasks/19

## Description

The data is a set of pixels in RGB format (each pixel is presented by 3 bytes). You need to send only the Blue component of each pixel to STDOUT.

## Input

- Binary data via STDIN (uint8 numbers in binary representation)
- Pixels in RGB format (3 bytes per pixel: R, G, B)
- Number of pixels: 150,000,000

## Output

- Write only the Blue component (3rd byte of each pixel) to STDOUT as binary data

## Notes

- This task is about efficient memory access and byte shuffling
- Optimal solutions use a load-shuffle-store pipeline (SIMD) to extract blue components
- Related tasks: "Blue from RGBA" and "Count uint8"

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
