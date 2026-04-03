"""Oracle for Task 15: Orderbook."""

import random
import sys
import time
from bisect import insort

N = 50_000
SEED = 42


def generate_data():
    """Generate valid orderbook operations by simulating the book."""
    rng = random.Random(SEED)
    operations = []
    # Simulate the book to ensure valid operations
    # orders: list of (price, seq, size), kept sorted by (price, seq)
    book = []
    seq = 0

    for _ in range(N):
        book_size = len(book)
        total_shares = sum(size for _, _, size in book)

        if book_size == 0:
            op_type = "insert"
        else:
            r = rng.random()
            if r < 0.58:
                op_type = "insert"
            elif r < 0.97:
                op_type = "delete"
            else:
                op_type = "buy" if total_shares > 0 else "insert"

        if op_type == "insert":
            price = rng.randint(800, 4100)
            size = rng.randint(20, 200)
            operations.append(("insert", price, size))
            insort(book, (price, seq, size))
            seq += 1

        elif op_type == "delete":
            position = rng.randint(0, book_size - 1)
            operations.append(("delete", position))
            book.pop(position)

        elif op_type == "buy":
            max_buy = min(total_shares, 100)
            quantity = rng.randint(1, max(1, max_buy))
            operations.append(("buy", quantity))
            # Simulate buy
            remaining = quantity
            while remaining > 0 and book:
                price, s, size = book[0]
                if size <= remaining:
                    remaining -= size
                    book.pop(0)
                else:
                    book[0] = (price, s, size - remaining)
                    remaining = 0

    return operations


def reference_solve(operations):
    orders = []  # list of (price, seq, size)
    seq = 0

    for op in operations:
        if op[0] == "insert":
            price, size = op[1], op[2]
            entry = (price, seq, size)
            seq += 1
            # Insert in sorted position (by price, then seq)
            lo, hi = 0, len(orders)
            while lo < hi:
                mid = (lo + hi) // 2
                if (orders[mid][0], orders[mid][1]) < (price, entry[1]):
                    lo = mid + 1
                else:
                    hi = mid
            orders.insert(lo, entry)

        elif op[0] == "delete":
            position = op[1]
            if position < len(orders):
                orders.pop(position)

        elif op[0] == "buy":
            quantity = op[1]
            while quantity > 0 and orders:
                price, s, size = orders[0]
                if size <= quantity:
                    quantity -= size
                    orders.pop(0)
                else:
                    orders[0] = (price, s, size - quantity)
                    quantity = 0

    return [f"{price},{size}" for price, s, size in orders]


def main():
    operations = generate_data()
    expected = reference_solve(operations)

    from solution import solve

    # Warmup with small subset
    warmup_ops = [("insert", 1000, 50), ("insert", 900, 30), ("buy", 10), ("delete", 0)]
    solve(warmup_ops)

    start = time.perf_counter()
    result = solve(operations)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {len(result)} orders, expected {len(expected)} orders", file=sys.stderr)
        for i in range(min(len(result), len(expected))):
            if result[i] != expected[i]:
                print(f"First mismatch at index {i}: got '{result[i]}', expected '{expected[i]}'", file=sys.stderr)
                break
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
