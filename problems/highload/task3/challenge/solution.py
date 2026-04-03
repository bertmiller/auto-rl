def solve(tokens: list[str]) -> int:
    unique = set()
    for t in tokens:
        unique.add(t)
    return len(unique)
