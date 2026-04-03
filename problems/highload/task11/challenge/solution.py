from datetime import datetime


def solve(timestamps: list[str]) -> int:
    total = 0
    for ts in timestamps:
        dt = datetime.fromisoformat(ts)
        total += int(dt.timestamp())
    return total
