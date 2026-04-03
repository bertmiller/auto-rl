def solve(numbers: list[str]) -> int:
    total = 0
    for s in numbers:
        total += int(s)
    return total & 0xFFFFFFFFFFFFFFFF
