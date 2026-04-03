"""Oracle for Task 11: RFC3339 Timestamps."""

import random
import sys
import time
from datetime import datetime, timezone, timedelta

N = 2_000_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    timestamps = []
    # Range: 1950-01-01 to 2050-12-31
    min_epoch = -631152000   # 1950-01-01T00:00:00 UTC
    max_epoch = 2556143999   # 2050-12-31T23:59:59 UTC
    offsets = list(range(-12 * 60, 14 * 60 + 1, 30))  # every 30 min from -12:00 to +14:00
    for _ in range(N):
        epoch = rng.randint(min_epoch, max_epoch)
        offset_minutes = rng.choice(offsets)
        tz = timezone(timedelta(minutes=offset_minutes))
        dt = datetime.fromtimestamp(epoch, tz=tz)
        timestamps.append(dt.isoformat())
    return timestamps


def reference_solve(timestamps):
    total = 0
    for ts in timestamps:
        dt = datetime.fromisoformat(ts)
        total += int(dt.timestamp())
    return total


def main():
    timestamps = generate_data()
    expected = reference_solve(timestamps)

    from solution import solve

    # Warmup
    solve(timestamps[:100])

    start = time.perf_counter()
    result = solve(timestamps)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
