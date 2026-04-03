import random
import struct
import math
import time
import sys

def is_prime(n):
    """Check if n is prime using trial division up to sqrt(n)."""
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

def generate_data(n=50_000, seed=42):
    """Generate n random uint32 values as bytes."""
    rng = random.Random(seed)
    numbers = [rng.getrandbits(32) for _ in range(n)]
    return struct.pack(f'<{n}I', *numbers)

def reference_solve(data):
    """Reference solution: check each number for primality and sum primes."""
    count = len(data) // 4
    numbers = struct.unpack(f'<{count}I', data)
    total = 0
    for n in numbers:
        if is_prime(n):
            total += n
    return total

def main():
    data = generate_data()

    # Compute expected answer with reference implementation
    expected = reference_solve(data)

    import solution

    # Timed run
    t0 = time.perf_counter()
    result = solution.solve(data)
    t1 = time.perf_counter()

    # Verify
    if not isinstance(result, int):
        print(f"ERROR: result is not int, got {type(result)}", file=sys.stderr)
        sys.exit(1)
    if result != expected:
        print(f"ERROR: expected {expected}, got {result}", file=sys.stderr)
        sys.exit(1)

    elapsed = t1 - t0
    print(f"time={elapsed:.6f}")

if __name__ == "__main__":
    main()
