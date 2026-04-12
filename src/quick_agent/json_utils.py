"""JSON parsing helpers."""

from __future__ import annotations


def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from text.
    This is a fallback for models that wrap JSON in extra text.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
        else:
            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise ValueError("Unbalanced JSON object in model output.")


def json_compatible_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        converted: dict[str, object] = {}
        for key, item in value.items():
            converted[str(key)] = json_compatible_value(item)
        return converted
    if isinstance(value, list):
        return [json_compatible_value(item) for item in value]
    if isinstance(value, tuple):
        return [json_compatible_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        if isinstance(payload, dict):
            return json_compatible_value(payload)
    return str(value)
