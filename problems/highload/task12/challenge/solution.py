import struct


def solve(data: bytes) -> str:
    count = len(data) // 4
    numbers = struct.unpack(f'<{count}I', data)
    parts = []
    for n in numbers:
        if n % 15 == 0:
            parts.append("FizzBuzz")
        elif n % 3 == 0:
            parts.append("Fizz")
        elif n % 5 == 0:
            parts.append("Buzz")
        else:
            parts.append(str(n))
    return "\n".join(parts) + "\n"
