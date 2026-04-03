import struct


def solve(data: bytes) -> int:
    n = len(data) // 4
    values = list(struct.unpack(f"<{n}I", data))
    values.sort()
    return values[n // 2]
