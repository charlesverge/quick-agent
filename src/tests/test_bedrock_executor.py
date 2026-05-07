import json
import os

import pytest

pytest.importorskip("boto3")
pytest.importorskip("botocore")
pytest.importorskip("types_boto3_bedrock")
pytest.importorskip("types_boto3_bedrock_runtime")
pytest.importorskip("types_boto3_s3")

import boto3

boto3.set_stream_logger('')

from pydantic import BaseModel

from quick_agent.agent_utils import parse_structured_result
from quick_agent.bedrock import BedrockExecutor


class StructuredOutput(BaseModel):
    title: str
    count: int

def _env_value(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        pytest.skip(f"{name} is required for Bedrock real API tests")
    assert value
    return value.strip()


def _executor(model_id: str = "qwen.qwen3-32b-v1:0") -> BedrockExecutor:
    return BedrockExecutor(
        region=_env_value("BEDROCK_BATCH_REGION"),
        role_arn=_env_value("BEDROCK_BATCH_ROLE_ARN"),
        model_id=model_id,
        s3_input_uri=_env_value("BEDROCK_BATCH_INPUT_S3_URI"),
        s3_output_uri=_env_value("BEDROCK_BATCH_OUTPUT_S3_URI"),
        aws_profile=os.environ.get("AWS_PROFILE"),
        poll_seconds=15,
    )


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "count": {"type": "number"},
        },
        "required": ["title", "count"],
        "additionalProperties": False,
    }


def _assert_structured_text(value: str) -> None:
    try:
        structured = parse_structured_result(value, StructuredOutput)
        assert isinstance(structured, StructuredOutput)
        assert structured.title == "alpha"
        assert structured.count == 7
        print(structured)
    except Exception as e:
        print(value)
        pytest.fail(f"Failed to parse structured output: {e}")


def _converse_input() -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Return JSON with title alpha and count 7.",
                    }
                ],
            }
        ],
        "outputConfig": {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "quick_agent_contract",
                        "description": "Quick Agent Bedrock contract test",
                        "schema": json.dumps(
                            _schema(), ensure_ascii=True, separators=(",", ":")
                        ),
                    }
                },
            }
        },
    }


def _open_weight_invoke_input() -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "content": "Return JSON with title alpha and count 7.",
            }
        ],
        "inferenceConfig": {
            "maxTokens": 256,
            "temperature": 0.1,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "quick_agent_contract",
                "schema": _schema(),
            },
        },
    }


def _anthropic_invoke_input() -> dict[str, object]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Return JSON with title alpha and count 7.",
                    }
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": _schema(),
            }
        },
    }


def _converse_text(response: dict[str, object]) -> str:
    output = response["output"]
    assert isinstance(output, dict)
    message = output["message"]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    first = content[0]
    assert isinstance(first, dict)
    text = first["text"]
    assert isinstance(text, str)
    return text


def _invoke_text(response: dict[str, object]) -> str:
    choices = response.get("choices")
    print(json.dumps(response, indent=2))
    if isinstance(choices, list):
        first = choices[0]
        assert isinstance(first, dict)
        message = first["message"]
        assert isinstance(message, dict)
        text = message["content"]
        assert isinstance(text, str)
        return text
    content = response["content"]
    assert isinstance(content, list)
    first = content[0]
    assert isinstance(first, dict)
    text = first["text"]
    assert isinstance(text, str)
    return text


def _batch_output_text(row: dict[str, object]) -> str:
    model_output = row["modelOutput"]
    assert isinstance(model_output, dict)
    if "output" in model_output:
        return _converse_text(model_output)
    return _invoke_text(model_output)


def _batch_rows(model_input: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    while index < 100:
        rows.append({"recordId": f"r{index}", "modelInput": model_input})
        index += 1
    return rows


def _record(rows: list[dict[str, object]], record_id: str) -> dict[str, object]:
    for row in rows:
        row_record_id = row.get("recordId")
        if row_record_id == record_id:
            return row
    raise ValueError(f"Batch output missing recordId={record_id}.")


def test_bedrock_converse_structured_ondemand_real_api() -> None:
    executor = _executor()
    response = executor.run_ondemand("converse", _converse_input())
    _assert_structured_text(_converse_text(response))


def test_bedrock_converse_structured_batch_real_api() -> None:
    executor = _executor()
    input_name = f"{executor.test_name('converse')}.jsonl"
    job_id = executor.test_name("qa-converse")
    output = executor.run_batch(
        "Converse", input_name, _batch_rows(_converse_input()), job_id
    )
    assert len(output) == 100
    _assert_structured_text(_batch_output_text(_record(output, "r0")))


def test_bedrock_open_weight_structured_ondemand_real_api() -> None:
    executor = _executor()
    response = executor.run_ondemand("invoke", _open_weight_invoke_input())
    _assert_structured_text(_invoke_text(response))


def test_bedrock_open_weight_structured_batch_real_api() -> None:
    executor = _executor()
    input_name = f"{executor.test_name('invoke')}.jsonl"
    job_id = executor.test_name("qa-invoke")
    output = executor.run_batch(
        "InvokeModel", input_name, _batch_rows(_open_weight_invoke_input()), job_id
    )
    assert len(output) == 100
    _assert_structured_text(_batch_output_text(_record(output, "r0")))

@pytest.mark.skip(reason="Anthropic invoke is not yet supported yet")
def test_bedrock_anthropic_structured_ondemand_real_api() -> None:
    if os.environ.get("BEDROCK_BATCH_RUN_ANTHROPIC_INVOKE") != "1":
        pytest.skip("BEDROCK_BATCH_RUN_ANTHROPIC_INVOKE=1 is required")
    executor = _executor(model_id="anthropic.claude-3-haiku-20240307-v1:0")
    response = executor.run_ondemand("invoke", _anthropic_invoke_input())
    _assert_structured_text(_invoke_text(response))
