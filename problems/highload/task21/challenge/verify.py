"""Oracle for Task 21: CRC of Numbers."""

import random
import struct
import sys
import time

N = 2_000_000
SEED = 42
MASK = 0xFFFFFFFFFFFFFFFF


def generate_data():
    rng = random.Random(SEED)
    values = [rng.randint(0, 0xFFFFFFFF) for _ in range(N)]
    return struct.pack(f'<{N}I', *values)


def reference_solve(data):
    n = len(data) // 4
    total = 0
    for i in range(n):
        num = struct.unpack_from('<I', data, i * 4)[0]
        s = str(num)
        crc = 0
        for pos, ch in enumerate(s):
            crc += int(ch) * (pos + 1)
        total += crc
    return total & MASK


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    warmup = struct.pack('<II', 123, 456)
    solve(warmup)

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
