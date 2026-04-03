"""Oracle for Task 12: FizzBuzz."""

import random
import struct
import sys
import time

N = 20_000_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    return rng.randbytes(N * 4)


def reference_solve(data):
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


def main():
    data = generate_data()
    expected = reference_solve(data)

    from solution import solve

    # Warmup
    solve(data[:400])

    start = time.perf_counter()
    result = solve(data)
    elapsed = time.perf_counter() - start

    if result != expected:
        # Show first difference
        result_lines = result.split('\n')
        expected_lines = expected.split('\n')
        for i, (r, e) in enumerate(zip(result_lines, expected_lines)):
            if r != e:
                print(f"WRONG at line {i}: got '{r}', expected '{e}'", file=sys.stderr)
                break
        else:
            print(f"WRONG: length mismatch got {len(result_lines)} vs {len(expected_lines)}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
