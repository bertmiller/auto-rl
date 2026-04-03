def solve(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result
