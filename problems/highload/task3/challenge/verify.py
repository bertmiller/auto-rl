"""Oracle for Task 3: Unique Strings."""

import random
import sys
import time

N_TOTAL = 20_000_000
N_UNIQUE = 2_000_000
MAX_LEN = 16
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@#%*"
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    # Generate unique tokens
    unique_tokens = set()
    while len(unique_tokens) < N_UNIQUE:
        length = rng.randint(1, MAX_LEN)
        token = "".join(rng.choices(CHARSET, k=length))
        unique_tokens.add(token)
    unique_list = list(unique_tokens)
    # Build full token list by sampling from unique tokens
    tokens = [rng.choice(unique_list) for _ in range(N_TOTAL)]
    return tokens


def reference_solve(tokens):
    return len(set(tokens))


def main():
    tokens = generate_data()
    expected = reference_solve(tokens)

    from solution import solve

    # Warmup
    solve(tokens[:100])

    start = time.perf_counter()
    result = solve(tokens)
    elapsed = time.perf_counter() - start

    if result != expected:
        print(f"WRONG: got {result}, expected {expected}", file=sys.stderr)
        sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
