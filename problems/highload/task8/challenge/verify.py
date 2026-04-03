import random
import struct
import time
import sys

def generate_data(n=5_000_000, seed=42):
    """Generate n random uint32 values as bytes."""
    rng = random.Random(seed)
    numbers = [rng.getrandbits(32) for _ in range(n)]
    return struct.pack(f'<{n}I', *numbers)

def reference_solve(data):
    """Reference solution: sort and sum top 100."""
    count = len(data) // 4
    numbers = struct.unpack(f'<{count}I', data)
    sorted_nums = sorted(numbers, reverse=True)
    return sum(sorted_nums[:100])

def main():
    data = generate_data()

    # Compute expected answer
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
