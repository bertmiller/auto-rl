# Task 15: Orderbook

**Source:** https://highload.fun/tasks/15

## Description

You need to maintain an order book and process a series of operations. Orders are kept sorted by price-time priority: orders with better (lower) prices take priority over higher prices; earlier orders take priority within the same price. Position 0 refers to the current best (lowest price) offer.

Once 1,000,000 updates are processed, buy 1,000 shares from the top of the book.

## Operations

Three types of operations are sent via STDIN:

1. **Insert** - Add an offer to the orderbook at a given price and size
2. **Delete** - Delete the nth order in the book (using price-time priority ordering)
3. **Buy** - Lift n lots from the orderbook

## Input Statistics (observed)

- Price range: ~815-4109 (typically 0-5000)
- Size range: 20-200
- Position range: 0-1014
- Share quantities for buys: 200-2000
- Maximum queue depth: consistently under 30
- Approximate operation ratio: ~59% inserts, ~39% deletes, ~2% buys
- Total updates: 1,000,000

## Output

- Process all operations and output results as required

## Constraints

- Optimize for speed (CPU time)
- Languages: C, C++, Go, Rust
