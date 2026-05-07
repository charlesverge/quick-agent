"""JSON parsing helpers."""

from __future__ import annotations

import json
import logging
import re

import json_repair
import openai

from .exceptions import QuickAgentLLMTemporaryException

logger = logging.getLogger(__name__)

_SUPPORTED_BEDROCK_JSON_TYPES = (
    "object",
    "array",
    "string",
    "integer",
    "number",
    "boolean",
    "null",
)
_SUPPORTED_BEDROCK_STRING_FORMATS = (
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
)
_UNSUPPORTED_BEDROCK_SCHEMA_KEYS = (
    "minimum",
    "maximum",
    "multipleOf",
    "minLength",
    "maxLength",
)
_JSON_POINTER_ROOT = "#"


def _is_supported_bedrock_literal(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, int, float)):
        return True
    return False


def _decode_json_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_json_pointer(
    schema: dict[str, object], pointer: str
) -> dict[str, object] | None:
    if pointer == _JSON_POINTER_ROOT:
        return schema
    if not pointer.startswith("#/"):
        return None
    current: object = schema
    for raw_token in pointer[2:].split("/"):
        token = _decode_json_pointer_token(raw_token)
        if isinstance(current, dict):
            current = current.get(token)
            continue
        if isinstance(current, list):
            if not token.isdigit():
                return None
            index = int(token)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    if isinstance(current, dict):
        return current
    return None


def _validate_bedrock_type(type_value: object, path: str) -> None:
    if isinstance(type_value, str):
        if type_value not in _SUPPORTED_BEDROCK_JSON_TYPES:
            raise ValueError(
                f"{path}: unsupported Bedrock JSON schema type {type_value!r}."
            )
        return
    if isinstance(type_value, list):
        if len(type_value) == 0:
            raise ValueError(f"{path}: type list must not be empty.")
        for item in type_value:
            if not isinstance(item, str):
                raise ValueError(f"{path}: type list entries must be strings.")
            if item not in _SUPPORTED_BEDROCK_JSON_TYPES:
                raise ValueError(
                    f"{path}: unsupported Bedrock JSON schema type {item!r}."
                )
        return
    raise ValueError(f"{path}: type must be a string or list of strings.")


def _validate_bedrock_enum(values: object, path: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{path}: enum must be a list.")
    for value in values:
        if not _is_supported_bedrock_literal(value):
            raise ValueError(
                f"{path}: enum values must be strings, numbers, booleans, or null."
            )


def _validate_bedrock_const(value: object, path: str) -> None:
    if not _is_supported_bedrock_literal(value):
        raise ValueError(f"{path}: const must be a string, number, boolean, or null.")


def _validate_bedrock_schema_node(
    node: object,
    *,
    path: str,
    root_schema: dict[str, object],
    ref_stack: list[str],
) -> None:
    if isinstance(node, list):
        index = 0
        while index < len(node):
            _validate_bedrock_schema_node(
                node[index],
                path=f"{path}[{index}]",
                root_schema=root_schema,
                ref_stack=ref_stack,
            )
            index += 1
        return
    if not isinstance(node, dict):
        return
    ref_value = node.get("$ref")
    if ref_value is not None:
        if not isinstance(ref_value, str):
            raise ValueError(f"{path}.$ref: $ref must be a string.")
        if not ref_value.startswith("#"):
            raise ValueError(f"{path}.$ref: external $ref values are not supported.")
        if ref_value in ref_stack:
            raise ValueError(f"{path}.$ref: recursive schemas are not supported.")
        resolved = _resolve_json_pointer(root_schema, ref_value)
        if resolved is None:
            raise ValueError(
                f"{path}.$ref: unresolved internal reference {ref_value!r}."
            )
        _validate_bedrock_schema_node(
            resolved,
            path=f"{path}.$ref({ref_value})",
            root_schema=root_schema,
            ref_stack=[*ref_stack, ref_value],
        )
    type_value = node.get("type")
    if type_value is not None:
        _validate_bedrock_type(type_value, f"{path}.type")
    enum_value = node.get("enum")
    if enum_value is not None:
        _validate_bedrock_enum(enum_value, f"{path}.enum")
    const_value = node.get("const")
    if const_value is not None:
        _validate_bedrock_const(const_value, f"{path}.const")
    format_value = node.get("format")
    if format_value is not None:
        if not isinstance(format_value, str):
            raise ValueError(f"{path}.format: format must be a string.")
        if format_value not in _SUPPORTED_BEDROCK_STRING_FORMATS:
            raise ValueError(
                f"{path}.format: unsupported Bedrock string format {format_value!r}."
            )
    min_items = node.get("minItems")
    if min_items is not None:
        if not isinstance(min_items, int) or isinstance(min_items, bool):
            raise ValueError(f"{path}.minItems: minItems must be an integer.")
        if min_items not in (0, 1):
            raise ValueError(
                f"{path}.minItems: Bedrock only supports minItems values 0 and 1."
            )
    additional_properties = node.get("additionalProperties")
    if "additionalProperties" in node and additional_properties is not False:
        raise ValueError(
            f"{path}.additionalProperties: additionalProperties must be false for Bedrock."
        )
    for key in _UNSUPPORTED_BEDROCK_SCHEMA_KEYS:
        if key in node:
            raise ValueError(f"{path}.{key}: {key} is not supported by Bedrock.")
    any_of = node.get("anyOf")
    if any_of is not None and not isinstance(any_of, list):
        raise ValueError(f"{path}.anyOf: anyOf must be a list.")
    all_of = node.get("allOf")
    if all_of is not None and not isinstance(all_of, list):
        raise ValueError(f"{path}.allOf: allOf must be a list.")
    for key, value in node.items():
        if key == "$ref":
            continue
        _validate_bedrock_schema_node(
            value,
            path=f"{path}.{key}",
            root_schema=root_schema,
            ref_stack=ref_stack,
        )


def validate_bedrock_schema(schema: dict[str, object], *, context: str) -> None:
    _validate_bedrock_schema_node(
        schema,
        path=context,
        root_schema=schema,
        ref_stack=[],
    )

_EXTRA_WS = r'(?:\s|\\n|\\r|\\t)*'

_RE_LEADING_EMPTY_ARRAY = re.compile(
    rf'^{_EXTRA_WS}\[\]{_EXTRA_WS}'
)

_RE_EXTRA_LEADING_BRACE = re.compile(
    rf'^\{{{_EXTRA_WS}(?:(?:\\")|")?{_EXTRA_WS}\{{{_EXTRA_WS}"?'
)

_RE_ESCAPED_OPENING_OBJECT_QUOTE = re.compile(
    rf'^\{{{_EXTRA_WS}\\\"'
)


def extract_extract_brackets(text: str) -> str:
    original = text

    # []{\"locations\":[]} -> {\"locations\":[]}
    repaired = _RE_LEADING_EMPTY_ARRAY.sub('', text, count=1)

    # {\"locations\":[]} -> {"locations":[]}
    repaired = _RE_ESCAPED_OPENING_OBJECT_QUOTE.sub('{"', repaired, count=1)

    # {"{"foo": "bar"} -> {"foo": "bar"}
    # {\"{"foo": "bar"} -> {"foo": "bar"}
    # {{"foo": "bar"} -> {"foo": "bar"}
    repaired = _RE_EXTRA_LEADING_BRACE.sub('{"', repaired, count=1)

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        raise QuickAgentLLMTemporaryException(
            message="extract_extract_brackets failed.",
            output=original,
        )

def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from text.
    This is a fallback for models that wrap JSON in extra text.
    """
    prefix = "QuickAgent.extract_first_json_object"
    start = text.find("{")
    if start == -1:
        raise ValueError(f"{prefix} No JSON object found in model output. Output was: {text!r}")

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
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    raise QuickAgentLLMTemporaryException(message="Unbalanced JSON object in model output.", output=text)

def repair_json_text(text: str, mode = 0) -> str:
    """
    Extract the first top-level JSON object from text.
    This is a fallback for models that wrap JSON in extra text.
    """
    if mode == 0:
      try:
        decoded_object = json_repair.loads(text)
        return json.dumps(decoded_object)
      except Exception:
          logger.error("f{prefix}: json_repair failed {text} {e}")
      return extract_first_json_object(text)
    if mode == 1:
      return extract_first_json_object(text)
    return extract_extract_brackets(text)
    


def json_compatible_value(value: object) -> object:
    if value is None:
        return None
    if value is openai.omit:
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
        payload = model_dump(mode="json", warnings="none", fallback=lambda _: None)
        return json_compatible_value(payload)
    return str(value)
