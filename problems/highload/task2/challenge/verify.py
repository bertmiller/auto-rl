"""Oracle for Task 2: XML to JSON."""

import json
import random
import sys
import time

N = 200_000
SEED = 42


def generate_data():
    rng = random.Random(SEED)
    parts = ['<people>']
    expected = []
    for i in range(1, N + 1):
        age = rng.randint(1, 100)
        height = round(rng.uniform(140.0, 210.0), 1)
        married = rng.choice([True, False])
        num_phones = rng.choices([0, 1, 2, 3], weights=[20, 40, 30, 10])[0]
        phones = []
        for _ in range(num_phones):
            phone = f"{rng.randint(100,999)}-{rng.randint(1000,9999)}"
            phones.append(phone)

        parts.append("<person>")
        parts.append(f"<id>{i}</id>")
        parts.append(f"<age>{age}</age>")
        parts.append(f"<height>{height}</height>")
        parts.append(f"<married>{'true' if married else 'false'}</married>")
        parts.append("<phones>")
        for ph in phones:
            parts.append(f"<phone>{ph}</phone>")
        parts.append("</phones>")
        parts.append("</person>")

        obj = {"id": i, "age": age, "height": height, "married": married}
        if phones:
            obj["phones"] = phones
        expected.append(obj)

    parts.append("</people>")
    xml_text = "".join(parts)
    return xml_text, expected


def main():
    xml_text, expected = generate_data()

    from solution import solve

    # Warmup
    solve("<people><person><id>1</id><age>25</age><height>175.5</height><married>true</married><phones></phones></person></people>")

    start = time.perf_counter()
    result_json = solve(xml_text)
    elapsed = time.perf_counter() - start

    try:
        result = json.loads(result_json)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"INVALID JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(result, list) or len(result) != len(expected):
        print(f"WRONG: expected {len(expected)} persons, got {len(result) if isinstance(result, list) else 'non-list'}", file=sys.stderr)
        sys.exit(1)

    for i, (got, exp) in enumerate(zip(result, expected)):
        if got != exp:
            print(f"WRONG at person {i}: got {got}, expected {exp}", file=sys.stderr)
            sys.exit(1)

    print(f"time={elapsed:.6f}")


if __name__ == "__main__":
    main()
