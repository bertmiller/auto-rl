# XML to JSON

## Problem

Convert an XML document containing person records into a JSON string.

## Function Signature

```python
def solve(xml_text: str) -> str:
```

### Parameters
- `xml_text`: An XML string in the following format:
```xml
<people>
  <person>
    <id>1</id>
    <age>25</age>
    <height>175.5</height>
    <married>true</married>
    <phones>
      <phone>555-1234</phone>
      <phone>555-5678</phone>
    </phones>
  </person>
  ...
</people>
```

### Returns
- A JSON string representing an array of person objects:
```json
[
  {
    "id": 1,
    "age": 25,
    "height": 175.5,
    "married": true,
    "phones": ["555-1234", "555-5678"]
  }
]
```

## Type Conversion Rules

- `id` and `age` are integers
- `height` is a float
- `married` is a boolean (`"true"` -> `true`, `"false"` -> `false`)
- `phones` is a list of strings. If the person has no phones, **omit the "phones" key entirely** from that person's object.

## Data Size

The input contains 200,000 person records.

## Notes

- The output must be valid JSON.
- Correctness is checked by parsing both the expected and actual JSON and comparing structurally (order of keys does not matter, but array order does).
- You may use any standard library (xml.etree, json, re) or third-party libraries available in the environment.
