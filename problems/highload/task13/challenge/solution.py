def solve(data: bytes) -> int:
    count = 0
    for b in data:
        if b == 127:
            count += 1
    return count
