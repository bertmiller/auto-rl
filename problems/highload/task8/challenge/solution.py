import struct

def solve(data: bytes) -> int:
    """Return the sum of the top 100 largest uint32 values."""
    count = len(data) // 4
    numbers = struct.unpack(f'<{count}I', data)
    sorted_nums = sorted(numbers, reverse=True)
    return sum(sorted_nums[:100])
