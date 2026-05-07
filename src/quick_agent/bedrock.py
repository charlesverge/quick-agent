"""Bedrock executor for real API contract checks."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal

from boto3.session import Session
from botocore.config import Config
from types_boto3_bedrock.client import BedrockClient
from types_boto3_bedrock_runtime import BedrockRuntimeClient
from types_boto3_bedrock_runtime.type_defs import (
    ContentBlockTypeDef,
    ConverseRequestTypeDef,
    JsonSchemaDefinitionTypeDef,
    MessageTypeDef,
    OutputConfigTypeDef,
    OutputFormatStructureTypeDef,
    OutputFormatTypeDef,
    ServiceTierTypeDef,
)
from types_boto3_s3.client import S3Client

JsonMap = dict[str, object]
OndemandInvocationType = Literal["converse", "invoke"]
BatchInvocationType = Literal["Converse", "InvokeModel"]


class BedrockExecutor:
    def __init__(
        self,
        *,
        region: str,
        role_arn: str,
        model_id: str,
        s3_input_uri: str,
        s3_output_uri: str,
        aws_profile: str | None = None,
        poll_seconds: int = 30,
        timeout_seconds: int = 60 * 60,
    ) -> None:
        self.region = region
        self.role_arn = role_arn
        self.model_id = model_id
        self.s3_input_uri = s3_input_uri
        self.s3_output_uri = s3_output_uri
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        config = Config(
            read_timeout=15 * 60,
            connect_timeout=60,
            retries={"max_attempts": 3},
        )
        if aws_profile is not None and aws_profile.strip() != "":
            session = Session(profile_name=aws_profile.strip(), region_name=region)
        else:
            session = Session(region_name=region)
        self.bedrock: BedrockClient = session.client("bedrock", config=config)
        self.bedrock_runtime: BedrockRuntimeClient = session.client(
            "bedrock-runtime", config=config
        )
        self.s3: S3Client = session.client("s3", config=config)

    def run_ondemand(
        self,
        invocation_type: OndemandInvocationType,
        model_input: JsonMap,
    ) -> JsonMap:
        if invocation_type == "converse":
            request = self._converse_request(model_input)
            response = self.bedrock_runtime.converse(**request)
            return self._json_map(response)
        invoke_response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(model_input, ensure_ascii=True, separators=(",", ":")),
        )
        body = invoke_response["body"].read()
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Bedrock invoke response body must decode to an object.")
        return self._json_map(parsed)

    def run_batch(
        self,
        invocation_type: BatchInvocationType,
        input_name: str,
        rows: list[JsonMap],
        job_id: str,
    ) -> list[JsonMap]:
        input_uri = self.upload_input(input_name=input_name, rows=rows)
        job_arn = self.submit_job(
            job_id=job_id,
            input_uri=input_uri,
            invocation_type=invocation_type,
        )
        self.wait_job(job_arn)
        aws_job_id = job_arn.split("/")[-1]
        return self.download_output(job_id=aws_job_id, input_name=input_name)

    def upload_input(
        self,
        *,
        input_name: str,
        rows: list[JsonMap],
    ) -> str:
        input_uri = self._compose_input_uri(input_name)
        bucket, key = self.parse_s3_uri(input_uri)
        body = ""
        for row in rows:
            body += json.dumps(row, ensure_ascii=True, separators=(",", ":"))
            body += "\n"
        self.s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
        return input_uri

    def submit_job(
        self,
        *,
        job_id: str,
        input_uri: str,
        invocation_type: BatchInvocationType,
    ) -> str:
        response = self.bedrock.create_model_invocation_job(
            modelId=self.model_id,
            roleArn=self.role_arn,
            jobName=job_id,
            inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": self.s3_output_uri}},
            modelInvocationType=invocation_type,
        )
        job_arn = response.get("jobArn")
        if not isinstance(job_arn, str) or job_arn == "":
            raise ValueError(
                "Bedrock create_model_invocation_job response missing jobArn."
            )
        return job_arn

    def wait_job(
        self,
        job_id: str,
    ) -> JsonMap:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            response = self.bedrock.get_model_invocation_job(jobIdentifier=job_id)
            status_obj = response.get("status")
            if not isinstance(status_obj, str):
                raise ValueError(
                    "Bedrock get_model_invocation_job response missing status."
                )
            if status_obj in ("Completed", "Failed", "Stopped"):
                if status_obj != "Completed":
                    raise ValueError(
                        f"Bedrock batch job ended with status={status_obj}."
                    )
                return self._json_map(response)
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"Bedrock batch job timed out job_id={job_id}.")

    def download_output(
        self,
        *,
        job_id: str,
        input_name: str,
    ) -> list[JsonMap]:
        output_uri = self.expected_output_uri(job_id=job_id, input_name=input_name)
        bucket, key = self.parse_s3_uri(output_uri)
        response = self.s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        rows: list[JsonMap] = []
        for line in body.splitlines():
            if line == "":
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError("Bedrock batch output line must decode to an object.")
            rows.append(self._json_map(parsed))
        return rows

    def expected_output_uri(
        self,
        *,
        job_id: str,
        input_name: str,
    ) -> str:
        output_uri = self.s3_output_uri
        if not output_uri.endswith("/"):
            output_uri = f"{output_uri}/"
        return f"{output_uri}{job_id}/{input_name}.out"

    def _compose_input_uri(
        self,
        input_name: str,
    ) -> str:
        input_uri = self.s3_input_uri
        if not input_uri.endswith("/"):
            input_uri = f"{input_uri}/"
        return f"{input_uri}{input_name}"

    @staticmethod
    def parse_s3_uri(
        uri: str,
    ) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid s3 uri uri={uri!r}.")
        value = uri[len("s3://") :]
        parts = value.split("/", 1)
        bucket = parts[0]
        key = "" if len(parts) == 1 else parts[1]
        if bucket == "":
            raise ValueError(f"S3 uri missing bucket uri={uri!r}.")
        return bucket, key

    @staticmethod
    def test_name(
        prefix: str,
    ) -> str:
        now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}-{now}-{int(time.time() * 1000)}"

    def _converse_request(
        self,
        model_input: JsonMap,
    ) -> ConverseRequestTypeDef:
        service_tier: ServiceTierTypeDef = {"type": "flex"}
        request: ConverseRequestTypeDef = {
            "modelId": self.model_id,
            "serviceTier": service_tier,
        }
        messages_obj = model_input.get("messages")
        if not isinstance(messages_obj, list):
            raise ValueError("Converse model_input.messages must be a list.")
        messages: list[MessageTypeDef] = []
        for item in messages_obj:
            if not isinstance(item, dict):
                raise ValueError("Converse message must be an object.")
            role_obj = item.get("role")
            if role_obj == "user":
                role: Literal["user", "assistant"] = "user"
            elif role_obj == "assistant":
                role = "assistant"
            else:
                raise ValueError("Converse message.role must be user or assistant.")
            content_obj = item.get("content")
            if not isinstance(content_obj, list):
                raise ValueError("Converse message.content must be a list.")
            content: list[ContentBlockTypeDef] = []
            for block_obj in content_obj:
                if not isinstance(block_obj, dict):
                    raise ValueError("Converse content block must be an object.")
                text_obj = block_obj.get("text")
                if not isinstance(text_obj, str):
                    raise ValueError("Converse content block.text must be a string.")
                content.append({"text": text_obj})
            messages.append({"role": role, "content": content})
        request["messages"] = messages
        output_config_obj = model_input.get("outputConfig")
        if output_config_obj is not None:
            request["outputConfig"] = self._output_config(output_config_obj)
        return request

    @staticmethod
    def _output_config(
        value: object,
    ) -> OutputConfigTypeDef:
        if not isinstance(value, dict):
            raise ValueError("Converse outputConfig must be an object.")
        text_format_obj = value.get("textFormat")
        if not isinstance(text_format_obj, dict):
            raise ValueError("Converse outputConfig.textFormat must be an object.")
        type_obj = text_format_obj.get("type")
        if type_obj != "json_schema":
            raise ValueError("Converse textFormat.type must be json_schema.")
        structure_obj = text_format_obj.get("structure")
        if not isinstance(structure_obj, dict):
            raise ValueError("Converse textFormat.structure must be an object.")
        json_schema_obj = structure_obj.get("jsonSchema")
        if not isinstance(json_schema_obj, dict):
            raise ValueError(
                "Converse textFormat.structure.jsonSchema must be an object."
            )
        schema_obj = json_schema_obj.get("schema")
        if not isinstance(schema_obj, str):
            raise ValueError("Converse jsonSchema.schema must be a string.")
        json_schema: JsonSchemaDefinitionTypeDef = {"schema": schema_obj}
        name_obj = json_schema_obj.get("name")
        if isinstance(name_obj, str):
            json_schema["name"] = name_obj
        description_obj = json_schema_obj.get("description")
        if isinstance(description_obj, str):
            json_schema["description"] = description_obj
        structure: OutputFormatStructureTypeDef = {"jsonSchema": json_schema}
        text_format: OutputFormatTypeDef = {
            "type": "json_schema",
            "structure": structure,
        }
        return {"textFormat": text_format}

    @staticmethod
    def _json_map(
        value: Mapping[str, object],
    ) -> JsonMap:
        result: JsonMap = {}
        for key, item in value.items():
            result[key] = item
        return result
