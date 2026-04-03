import random
import uuid
import time
import sys

def generate_data(n=5_000_000, seed=42):
    """Generate n random UUIDs deterministically."""
    rng = random.Random(seed)
    uuids = []
    for _ in range(n):
        # Generate UUID from a single random 128-bit int (much faster)
        bits = rng.getrandbits(128)
        u = uuid.UUID(int=bits, version=4)
        uuids.append(str(u))
    return uuids

def reference_solve(uuids):
    """Reference solution: simple sorted."""
    return sorted(uuids)

def main():
    uuids = generate_data()
    expected = reference_solve(uuids)

    import solution

    # Warm up
    solution.solve(uuids[:100])

    # Timed run
    t0 = time.perf_counter()
    result = solution.solve(uuids)
    t1 = time.perf_counter()

    # Verify
    if not isinstance(result, list):
        print("ERROR: result is not a list", file=sys.stderr)
        sys.exit(1)
    if len(result) != len(expected):
        print(f"ERROR: expected {len(expected)} items, got {len(result)}", file=sys.stderr)
        sys.exit(1)
    if result != expected:
        # Find first mismatch
        for i, (a, b) in enumerate(zip(result, expected)):
            if a != b:
                print(f"ERROR: mismatch at index {i}: got {a!r}, expected {b!r}", file=sys.stderr)
                break
        sys.exit(1)

    elapsed = t1 - t0
    print(f"time={elapsed:.6f}")

if __name__ == "__main__":
    main()
