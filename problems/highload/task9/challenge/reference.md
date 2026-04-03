# Task 9: Arithmetic Expression Evaluation

## Problem

Evaluate arithmetic expressions containing integers, the operators `+`, `-`, `*`, and parentheses. Return the results as a list of integers.

## Interface

```python
def solve(expressions: list[str]) -> list[int]:
    """Evaluate arithmetic expressions.

    Args:
        expressions: List of arithmetic expression strings.
            Each expression uses:
            - Integer numbers in int16 range (-32768 to 32767)
            - Operators: +, -, *
            - Parentheses for grouping
            - Standard operator precedence: * before + and -

    Returns:
        List of integer results, one per expression.
    """
```

## Constraints

- 1000 expressions, each containing ~2000 numbers and operators
- Numbers are in int16 range (-32768 to 32767)
- Expressions are well-formed (balanced parentheses, valid syntax)

## Example

```python
>>> solve(["3 + 5 * 2", "(3 + 5) * 2", "10 - 3 * 2"])
[13, 16, 4]
```
