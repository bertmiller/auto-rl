# Task 2: XML to JSON

**Source:** https://highload.fun/tasks/2

## Description

Your task is to convert persons from XML to JSON format for the shortest time.

## Input

- XML data via STDIN containing person records
- Number of persons: 1,000,000
- Each person has fields: id, age, height, married (boolean), and phones (array)
- Maximum of 3 phones per person

## Output

- JSON representation of the persons to STDOUT
- Preserve the order of persons
- Do not print the `phones` field if the array of phones is empty

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
