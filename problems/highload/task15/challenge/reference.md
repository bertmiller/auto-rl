# Orderbook

## Problem

Process a sequence of orderbook operations and return the remaining orders after all operations are complete.

The orderbook maintains orders sorted by price ascending (lower price = higher priority), with FIFO ordering for orders at the same price.

## Function Signature

```python
def solve(operations: list[tuple]) -> list[str]:
```

### Parameters
- `operations`: A list of 50,000 tuples, each one of:
  - `("insert", price, size)` -- Insert an order with the given price (int) and size (int) into the book.
  - `("delete", position)` -- Delete the order at the given position (0-indexed from best/lowest price). Position 0 is the highest-priority order.
  - `("buy", quantity)` -- Buy `quantity` shares from the top of the book. Consumes orders starting from position 0. If an order has more shares than needed, its size is reduced. If an order is fully consumed, it is removed.

### Returns
- A list of strings, one per remaining order, in priority order (ascending price, FIFO within same price). Each string is formatted as `"price,size"`.

## Example

```python
>>> ops = [
...     ("insert", 100, 50),
...     ("insert", 100, 30),
...     ("insert", 200, 40),
...     ("buy", 60),
...     ("delete", 0),
... ]
>>> solve(ops)
['200,40']
```

Explanation:
1. Insert (100, 50) -> book: [(100,50)]
2. Insert (100, 30) -> book: [(100,50), (100,30)]
3. Insert (200, 40) -> book: [(100,50), (100,30), (200,40)]
4. Buy 60: consume (100,50) fully (10 left to buy), consume 10 from (100,30) -> book: [(100,20), (200,40)]
5. Delete position 0: remove (100,20) -> book: [(200,40)]

## Notes

- Approximately 29,000 inserts, 19,500 deletes, and 1,500 buys in the 50,000 operations.
- Prices range from 800 to 4100, sizes from 20 to 200.
- Delete positions are always valid (< current book size).
- Buy quantities are always <= total available shares in the book.
- You may use any standard library for optimization (sortedcontainers, heapq, etc.).
