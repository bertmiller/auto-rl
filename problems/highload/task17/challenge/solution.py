import hashlib


def solve(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
