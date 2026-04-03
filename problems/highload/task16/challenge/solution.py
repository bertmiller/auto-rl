def solve(a_bytes: bytes, b_bytes: bytes) -> bytes:
    a = int.from_bytes(a_bytes, byteorder="little")
    b = int.from_bytes(b_bytes, byteorder="little")
    product = a * b
    result_len = len(a_bytes) + len(b_bytes)
    return product.to_bytes(result_len, byteorder="little")
