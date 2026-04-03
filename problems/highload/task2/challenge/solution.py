import json
import xml.etree.ElementTree as ET


def solve(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    result = []
    for person in root.findall("person"):
        obj = {}
        obj["id"] = int(person.find("id").text)
        obj["age"] = int(person.find("age").text)
        obj["height"] = float(person.find("height").text)
        obj["married"] = person.find("married").text == "true"
        phones_elem = person.find("phones")
        phones = []
        if phones_elem is not None:
            for phone in phones_elem.findall("phone"):
                phones.append(phone.text)
        if phones:
            obj["phones"] = phones
        result.append(obj)
    return json.dumps(result)
