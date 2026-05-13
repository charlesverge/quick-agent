import json

import openai
import pytest
from pydantic import BaseModel

from quick_agent.json_utils import (
    extract_extract_brackets,
    extract_first_json_object,
    json_compatible_value,
    repair_json_text,
    validate_bedrock_schema,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{{"foo": "bar"}', '{"foo": "bar"}'),
        ('{"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{\"{"foo": "bar"}', '{"foo": "bar"}'),
        ('{{a": 1}', '{"a": 1}'),
        ('{"{a": 1}', '{"a": 1}'),
        ('{{a": 1}', '{"a": 1}'),
        ('{\n{a": 1}', '{"a": 1}'),
        ('{\n"{a": 1}', '{"a": 1}'),
        ('{\n{"foo": "bar"}', '{"foo": "bar"}'),
        ('{\n"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{\n\"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{"{"selected_source_urls":["http"]}', '{"selected_source_urls":["http"]}'),
        (r'[]{"locations":[]}', '{"locations":[]}'),
    ],
)
def test_extract_extract_brackets_handles_extra_bracket_variants(
    text: str, expected: str
) -> None:
    result = extract_extract_brackets(text)

    assert result == expected


def test_validate_bedrock_schema_rejects_minimum_keyword() -> None:
    with pytest.raises(ValueError, match="minimum is not supported"):
        validate_bedrock_schema({"type": "integer", "minimum": 0}, context="t")


def test_validate_bedrock_schema_rejects_maximum_keyword() -> None:
    with pytest.raises(ValueError, match="maximum is not supported"):
        validate_bedrock_schema({"type": "integer", "maximum": 100}, context="t")


def test_validate_bedrock_schema_rejects_multiple_of_keyword() -> None:
    with pytest.raises(ValueError, match="multipleOf is not supported"):
        validate_bedrock_schema({"type": "integer", "multipleOf": 5}, context="t")


def test_validate_bedrock_schema_rejects_min_length_keyword() -> None:
    with pytest.raises(ValueError, match="minLength is not supported"):
        validate_bedrock_schema({"type": "string", "minLength": 1}, context="t")


def test_validate_bedrock_schema_rejects_max_length_keyword() -> None:
    with pytest.raises(ValueError, match="maxLength is not supported"):
        validate_bedrock_schema({"type": "string", "maxLength": 100}, context="t")


def test_validate_bedrock_schema_rejects_external_ref() -> None:
    with pytest.raises(ValueError, match=r"external \$ref values are not supported"):
        validate_bedrock_schema({"$ref": "https://example.com/other.json"}, context="t")


def test_validate_bedrock_schema_accepts_internal_ref_resolution() -> None:
    validate_bedrock_schema(
        {
            "type": "object",
            "$defs": {"Name": {"type": "string"}},
            "properties": {"name": {"$ref": "#/$defs/Name"}},
        },
        context="t",
    )


def test_validate_bedrock_schema_rejects_recursive_internal_ref() -> None:
    with pytest.raises(ValueError, match="recursive schemas are not supported"):
        validate_bedrock_schema(
            {
                "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
                "$ref": "#/$defs/Node",
            },
            context="t",
        )


def test_validate_bedrock_schema_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match="unsupported Bedrock string format"):
        validate_bedrock_schema({"type": "string", "format": "phone"}, context="t")


@pytest.mark.parametrize(
    "format_value",
    [
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
    ],
)
def test_validate_bedrock_schema_accepts_supported_formats(format_value: str) -> None:
    validate_bedrock_schema({"type": "string", "format": format_value}, context="t")


def test_validate_bedrock_schema_rejects_min_items_value_two() -> None:
    with pytest.raises(ValueError, match="Bedrock only supports minItems values 0 and 1"):
        validate_bedrock_schema({"type": "array", "minItems": 2}, context="t")


@pytest.mark.parametrize("min_items", [0, 1])
def test_validate_bedrock_schema_accepts_min_items_zero_and_one(min_items: int) -> None:
    validate_bedrock_schema({"type": "array", "minItems": min_items}, context="t")


def test_validate_bedrock_schema_rejects_additional_properties_true() -> None:
    with pytest.raises(ValueError, match="additionalProperties must be false"):
        validate_bedrock_schema({"type": "object", "properties": {}, "additionalProperties": True}, context="t")


def test_validate_bedrock_schema_rejects_unsupported_type_string() -> None:
    with pytest.raises(ValueError, match="unsupported Bedrock JSON schema type"):
        validate_bedrock_schema({"type": "foobar"}, context="t")


def test_validate_bedrock_schema_rejects_enum_with_object_value() -> None:
    with pytest.raises(ValueError, match="enum values must be strings, numbers, booleans, or null"):
        validate_bedrock_schema({"enum": [{"obj": 1}]}, context="t")


def test_validate_bedrock_schema_rejects_const_with_object_value() -> None:
    with pytest.raises(ValueError, match="const must be a string, number, boolean, or null"):
        validate_bedrock_schema({"const": {"obj": 1}}, context="t")


def test_validate_bedrock_schema_accepts_const_with_null() -> None:
    validate_bedrock_schema({"const": None}, context="t")


def test_repair_json_text_mode_0_uses_json_repair_on_valid_garbage() -> None:
    result = repair_json_text('{"x": 1, "y": 2,}', mode=0)
    assert json.loads(result) == {"x": 1, "y": 2}


def test_repair_json_text_mode_0_falls_back_to_extract_first_json_object() -> None:
    result = repair_json_text('preface {"x":1} suffix', mode=0)
    assert json.loads(result) == {"x": 1}


def test_repair_json_text_mode_1_uses_extract_first_json_object() -> None:
    result = repair_json_text('preface {"a":"b"} trailing', mode=1)
    assert json.loads(result) == {"a": "b"}


def test_repair_json_text_mode_2_uses_extract_extract_brackets() -> None:
    result = repair_json_text('{{"foo":"bar"}', mode=2)
    assert json.loads(result) == {"foo": "bar"}


def test_extract_first_json_object_handles_nested_objects() -> None:
    result = extract_first_json_object('preamble {"a":{"b":{"c":1}}} trailing')
    assert json.loads(result) == {"a": {"b": {"c": 1}}}


def test_extract_first_json_object_handles_strings_with_braces() -> None:
    result = extract_first_json_object('{"key":"value with } brace"}')
    assert json.loads(result) == {"key": "value with } brace"}


def test_extract_first_json_object_handles_escaped_quotes_in_strings() -> None:
    result = extract_first_json_object('{"key":"value with \\\"quote\\\""}')
    assert json.loads(result) == {"key": "value with \"quote\""}


def test_extract_first_json_object_raises_temporary_exception_on_unbalanced() -> None:
    with pytest.raises(Exception) as exc_info:
        extract_first_json_object('{"unclosed":')
    assert hasattr(exc_info.value, "output")


def test_extract_first_json_object_raises_when_no_object_found() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_first_json_object("no json here at all")


def test_json_compatible_value_handles_basemodel() -> None:
    class Example(BaseModel):
        x: int

    result = json_compatible_value(Example(x=1))
    assert result == {"x": 1}


def test_json_compatible_value_converts_openai_omit_to_none() -> None:
    assert json_compatible_value(openai.omit) is None


def test_json_compatible_value_recurses_into_nested_dicts_lists_tuples() -> None:
    class Example(BaseModel):
        x: int

    source = {"a": [Example(x=1), openai.omit, (1, 2)], "b": {"nested": Example(x=2)}}
    result = json_compatible_value(source)
    assert result["a"][0] == {"x": 1}
    assert result["a"][1] is None
    assert result["a"][2] == [1, 2]
    assert result["b"]["nested"] == {"x": 2}


def test_json_compatible_value_converts_unknown_to_str() -> None:
    class Custom:
        def __str__(self) -> str:
            return "custom"

    assert json_compatible_value(Custom()) == "custom"


def test_json_compatible_value_passes_primitives_unchanged() -> None:
    assert json_compatible_value(42) == 42
    assert json_compatible_value("hello") == "hello"
    assert json_compatible_value(True) is True
    assert json_compatible_value(None) is None
    assert json_compatible_value(3.14) == 3.14
