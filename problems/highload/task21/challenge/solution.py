import struct


def solve(data: bytes) -> int:
    n = len(data) // 4
    total = 0
    for i in range(n):
        num = struct.unpack_from('<I', data, i * 4)[0]
        s = str(num)
        crc = 0
        for pos, ch in enumerate(s):
            crc += int(ch) * (pos + 1)
        total += crc
    return total & 0xFFFFFFFFFFFFFFFF
