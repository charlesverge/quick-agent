from __future__ import annotations

import time

import pytest

from quick_agent.bedrock import BedrockExecutor


class FakeBody:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self) -> bytes:
        return self._content


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], content: bytes, request: object | None = None) -> None:
        self.status_code = status
        self.headers = headers
        self._content = content
        self.request = request

    async def aread(self) -> bytes:
        return self._content


class FakeRuntime:
    def __init__(self) -> None:
        self.last_converse: dict[str, object] | None = None
        self.last_invoke: dict[str, object] | None = None
        self.converse_response: object | None = None
        self.invoke_response: object | None = None

    def converse(self, **kwargs: object) -> object:
        self.last_converse = kwargs
        assert self.converse_response is not None
        return self.converse_response

    def invoke_model(self, **kwargs: object) -> object:
        self.last_invoke = kwargs
        assert self.invoke_response is not None
        return self.invoke_response


class FakeS3:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []
        self.get_object_result: dict[str, object] | None = None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "Body": Body})

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert self.get_object_result is not None
        return self.get_object_result


class FakeBedrock:
    def __init__(self) -> None:
        self.jobs: list[dict[str, object]] = []
        self.next_response: dict[str, object] | None = None

    def get_model_invocation_job(self, *, jobIdentifier: str) -> dict[str, object]:
        assert self.next_response is not None
        return self.next_response


def _make_executor() -> BedrockExecutor:
    executor = object.__new__(BedrockExecutor)
    executor.region = "us-east-1"
    executor.role_arn = "role"
    executor.model_id = "model"
    executor.s3_input_uri = "s3://bucket/in"
    executor.s3_output_uri = "s3://bucket/out"
    executor.poll_seconds = 0
    executor.timeout_seconds = 5
    executor.bedrock = FakeBedrock()
    executor.bedrock_runtime = FakeRuntime()
    executor.s3 = FakeS3()
    return executor


def test_parse_s3_uri_with_bucket_only() -> None:
    bucket, key = BedrockExecutor.parse_s3_uri("s3://my-bucket")
    assert bucket == "my-bucket"
    assert key == ""


def test_parse_s3_uri_with_bucket_and_key() -> None:
    bucket, key = BedrockExecutor.parse_s3_uri("s3://my-bucket/some/key/path.jsonl")
    assert bucket == "my-bucket"
    assert key == "some/key/path.jsonl"


def test_parse_s3_uri_rejects_missing_scheme() -> None:
    with pytest.raises(ValueError, match="Invalid s3 uri"):
        BedrockExecutor.parse_s3_uri("my-bucket/path")


def test_parse_s3_uri_rejects_empty_bucket() -> None:
    with pytest.raises(ValueError, match="S3 uri missing bucket"):
        BedrockExecutor.parse_s3_uri("s3:///key/only")


def test_expected_output_uri_appends_job_id_and_input_name_suffix() -> None:
    executor = _make_executor()
    result = executor.expected_output_uri(job_id="job-123", input_name="input.jsonl")
    assert result == "s3://bucket/out/job-123/input.jsonl.out"


def test_expected_output_uri_adds_trailing_slash_when_missing() -> None:
    executor = _make_executor()
    executor.s3_output_uri = "s3://bucket/out"
    result = executor.expected_output_uri(job_id="j", input_name="i.jsonl")
    assert result == "s3://bucket/out/j/i.jsonl.out"


def test_converse_request_serializes_messages_with_user_role() -> None:
    executor = _make_executor()
    request = executor._converse_request({"messages": [{"role": "user", "content": [{"text": "hello"}]}]})
    assert request["modelId"] == "model"
    assert request["serviceTier"] == {"type": "flex"}
    assert request["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]


def test_converse_request_rejects_non_user_assistant_role() -> None:
    executor = _make_executor()
    with pytest.raises(ValueError, match="Converse message.role must be user or assistant."):
        executor._converse_request({"messages": [{"role": "system", "content": [{"text": "x"}]}]})


def test_converse_request_includes_output_config_json_schema() -> None:
    executor = _make_executor()
    request = executor._converse_request(
        {
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {"schema": '{"type":"object"}', "name": "X"}
                    },
                }
            },
        }
    )
    assert request["outputConfig"]["textFormat"]["type"] == "json_schema"
    assert request["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["name"] == "X"


def test_converse_request_rejects_non_json_schema_text_format() -> None:
    executor = _make_executor()
    with pytest.raises(ValueError, match="Converse textFormat.type must be json_schema."):
        executor._converse_request(
            {
                "messages": [{"role": "user", "content": [{"text": "hello"}]}],
                "outputConfig": {"textFormat": {"type": "text"}},
            }
        )


def test_converse_request_rejects_missing_messages_list() -> None:
    executor = _make_executor()
    with pytest.raises(ValueError, match="Converse model_input.messages must be a list."):
        executor._converse_request({})


def test_converse_request_rejects_non_string_content_text() -> None:
    executor = _make_executor()
    with pytest.raises(ValueError, match="Converse content block.text must be a string."):
        executor._converse_request({"messages": [{"role": "user", "content": [{"text": 123}]}]})


@pytest.mark.usefixtures("monkeypatch")
def test_wait_job_returns_when_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _make_executor()
    executor.bedrock.next_response = {"status": "Completed", "outputDataConfig": {}}
    monkeypatch.setattr(time, "sleep", lambda _: None)
    response = executor.wait_job("job-arn")
    assert response == {"status": "Completed", "outputDataConfig": {}}


@pytest.mark.usefixtures("monkeypatch")
def test_wait_job_polls_until_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = _make_executor()
    executor.bedrock.next_response = {"status": "InProgress"}
    called = {"count": 0}

    def fake_get(**kwargs: object) -> dict[str, object]:
        called["count"] += 1
        return {"status": "Completed", "outputDataConfig": {}} if called["count"] >= 3 else {"status": "InProgress"}

    executor.bedrock.get_model_invocation_job = fake_get
    sleep_calls = {"count": 0}

    def fake_sleep(_: object) -> None:
        sleep_calls["count"] += 1

    monkeypatch.setattr(time, "sleep", fake_sleep)
    result = executor.wait_job("job-arn")
    assert result["status"] == "Completed"
    assert called["count"] == 3
    assert sleep_calls["count"] == 2


def test_wait_job_raises_on_failed_status() -> None:
    executor = _make_executor()
    executor.bedrock.next_response = {"status": "Failed"}
    with pytest.raises(ValueError, match="Bedrock batch job ended with status=Failed."):
        executor.wait_job("job-arn")


def test_wait_job_raises_on_stopped_status() -> None:
    executor = _make_executor()
    executor.bedrock.next_response = {"status": "Stopped"}
    with pytest.raises(ValueError, match="Bedrock batch job ended with status=Stopped."):
        executor.wait_job("job-arn")


def test_wait_job_raises_on_missing_status_field() -> None:
    executor = _make_executor()
    executor.bedrock.next_response = {}
    with pytest.raises(ValueError, match="Bedrock get_model_invocation_job response missing status."):
        executor.wait_job("job-arn")


def test_upload_input_serializes_rows_as_jsonl() -> None:
    executor = _make_executor()
    executor.s3.put_object_result = None
    rows = [{"recordId": "r1", "x": 1}, {"recordId": "r2", "x": 2}]
    executor.upload_input(input_name="test.jsonl", rows=rows)
    assert executor.s3.put_calls[0]["Bucket"] == "bucket"
    assert executor.s3.put_calls[0]["Key"] == "in/test.jsonl"
    assert executor.s3.put_calls[0]["Body"] == b'{"recordId":"r1","x":1}\n{"recordId":"r2","x":2}\n'


def test_upload_input_appends_input_name_to_s3_uri() -> None:
    executor = _make_executor()
    executor.s3_input_uri = "s3://bucket/in"
    executor.upload_input(input_name="jobs/batch.jsonl", rows=[{"recordId": "r1"}])
    assert executor.s3.put_calls[0]["Bucket"] == "bucket"
    assert executor.s3.put_calls[0]["Key"] == "in/jobs/batch.jsonl"


def test_upload_input_returns_full_s3_uri() -> None:
    executor = _make_executor()
    result = executor.upload_input(input_name="input.jsonl", rows=[{"recordId": "r1"}])
    assert result == "s3://bucket/in/input.jsonl"


def test_download_output_parses_jsonl_lines() -> None:
    executor = _make_executor()
    executor.s3.get_object_result = {"Body": FakeBody(b'{"a":1}\n{"b":2}\n{"c":3}\n')}
    rows = executor.download_output(job_id="job", input_name="input")
    assert rows == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_download_output_rejects_non_object_lines() -> None:
    executor = _make_executor()
    executor.s3.get_object_result = {"Body": FakeBody(b'["array","not","object"]\n')}
    with pytest.raises(ValueError, match="Bedrock batch output line must decode to an object."):
        executor.download_output(job_id="job", input_name="input")


def test_download_output_skips_empty_lines() -> None:
    executor = _make_executor()
    executor.s3.get_object_result = {"Body": FakeBody(b'{"a":1}\n\n{"b":2}\n')}
    rows = executor.download_output(job_id="job", input_name="input")
    assert rows == [{"a": 1}, {"b": 2}]


def test_download_output_decodes_utf8() -> None:
    executor = _make_executor()
    executor.s3.get_object_result = {"Body": FakeBody("{\"a\":\"é\"}\n".encode("utf-8"))}
    rows = executor.download_output(job_id="job", input_name="input")
    assert rows == [{"a": "é"}]


def test_run_ondemand_converse_calls_converse_endpoint() -> None:
    executor = _make_executor()
    executor.bedrock_runtime.converse_response = {"result": "ok"}
    response = executor.run_ondemand("converse", {"messages": [{"role": "user", "content": [{"text": "hello"}]}]})
    assert response == {"result": "ok"}
    assert executor.bedrock_runtime.last_converse is not None


def test_run_ondemand_invoke_decodes_body_bytes() -> None:
    executor = _make_executor()
    executor.bedrock_runtime.invoke_response = {"body": FakeBody(b'{"x":1}')}
    response = executor.run_ondemand("invoke", {"messages": []})
    assert response == {"x": 1}
    assert executor.bedrock_runtime.last_invoke is not None


def test_run_ondemand_invoke_rejects_non_object_body() -> None:
    executor = _make_executor()
    executor.bedrock_runtime.invoke_response = {"body": FakeBody(b'[1,2,3]')}
    with pytest.raises(ValueError, match="Bedrock invoke response body must decode to an object."):
        executor.run_ondemand("invoke", {"messages": []})
