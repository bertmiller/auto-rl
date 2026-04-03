"""Verify and score a difference basis construction."""


def verify_and_score():
    from solution import solve

    result = solve()

    # Validate format
    if not isinstance(result, (list, tuple)):
        print("score=-1")
        return

    B = list(set(int(x) for x in result))  # deduplicate and convert to int

    if len(B) < 2:
        print("score=-1")
        return

    # Check all elements are non-negative
    if any(x < 0 for x in B):
        print("score=-1")
        return

    B.sort()

    # Compute all positive differences
    diffs = set()
    for i in range(len(B)):
        for j in range(i):
            diffs.add(B[i] - B[j])

    # Find the largest n such that {1, ..., n} is contained in diffs
    n = 0
    k = 1
    while k in diffs:
        n = k
        k += 1

    if n == 0:
        print("score=-1")
        return

    size_B = len(B)
    score = n / (size_B * size_B)

    print(f"score={score:.10f}")


if __name__ == "__main__":
    verify_and_score()
