def solve(operations: list[tuple]) -> list[str]:
    # orders is kept sorted by (price, insertion_order)
    orders = []  # list of (price, size, seq)
    seq = 0

    for op in operations:
        if op[0] == "insert":
            price, size = op[1], op[2]
            # Insert in sorted position
            entry = (price, size, seq)
            seq += 1
            # Linear insertion to maintain sorted order
            inserted = False
            for i in range(len(orders)):
                if (price, entry[2]) < (orders[i][0], orders[i][2]):
                    orders.insert(i, entry)
                    inserted = True
                    break
            if not inserted:
                orders.append(entry)

        elif op[0] == "delete":
            position = op[1]
            orders.pop(position)

        elif op[0] == "buy":
            quantity = op[1]
            while quantity > 0 and orders:
                price, size, s = orders[0]
                if size <= quantity:
                    quantity -= size
                    orders.pop(0)
                else:
                    orders[0] = (price, size - quantity, s)
                    quantity = 0

    return [f"{price},{size}" for price, size, s in orders]
