from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, AliasChoices


class Plan(BaseModel):
    # LLMs sometimes emit structured steps; allow both strings and dicts.
    steps: list[str | dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[str | dict[str, Any]] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    field: str
    reason: str


class ValidationResult(BaseModel):
    valid: bool
    function_name: str | None = None
    parameters_count: int | None = None
    return_type: str | None = None
    target_path: str | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    summary: str | None = None


class BusinessSummary(BaseModel):
    company_name: str
    location: str
    summary: str


class EvalRowResult(BaseModel):
    index: int = 0
    agent: str
    command_file: str
    result: str = Field(default="", validation_alias=AliasChoices("result", "results"))  # "PASS" or "FAIL"
    note: str | None = None


class EvalListResult(BaseModel):
    total: int
    passed: int
    failed: int
    results_path: str
    rows: list[EvalRowResult] = Field(default_factory=list)


class ContainsCheck(BaseModel):
    text: str
    found: bool


class ContainsValidationResult(BaseModel):
    status: str  # "PASS" or "FAIL"
    checks: list[ContainsCheck] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    eval_path: str | None = None
    response_path: str | None = None
