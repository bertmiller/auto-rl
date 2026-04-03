def solve(data: bytes) -> bytes:
    result = bytearray()
    for i in range(2, len(data), 3):
        result.append(data[i])
    return bytes(result)
