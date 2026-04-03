import struct
import math

def is_prime(n):
    """Check if n is prime using trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def solve(data: bytes) -> int:
    """Sum all prime numbers in the binary data."""
    count = len(data) // 4
    numbers = struct.unpack(f'<{count}I', data)
    total = 0
    for n in numbers:
        if is_prime(n):
            total += n
    return total
