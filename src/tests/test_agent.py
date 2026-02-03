import json
import sys
import types
from pathlib import Path

import pytest

from pydantic import BaseModel

from quick_agent import agent_registry
from quick_agent import io_utils
from quick_agent import prompting
from quick_agent import tools_loader
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models import AgentSpec
from quick_agent.models import ChainStepSpec
from quick_agent.models import LoadedAgentFile
from quick_agent.models import ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.quick_agent import resolve_schema


def test_import_symbol_valid_and_invalid() -> None:
    tmp_module = types.ModuleType("tmpmod")
    tmp_module.__dict__["Value"] = 123
    sys.modules["tmpmod"] = tmp_module
    try:
        assert tools_loader.import_symbol("tmpmod:Value") == 123
        with pytest.raises(ValueError):
            tools_loader.import_symbol("tmpmod.Value")
    finally:
        sys.modules.pop("tmpmod", None)


def test_split_step_sections_extracts_content() -> None:
    body = """
# Title

## step:one

Hello one.

## step:two

Hello two.
"""
    sections = agent_registry.split_step_sections(body)
    assert sections["step:one"] == "Hello one."
    assert sections["step:two"] == "Hello two."


def test_load_input_json_and_text(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    json_path = safe_root / "in.json"
    json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    run_input = io_utils.load_input(json_path, permissions)
    assert run_input.kind == "json"
    assert run_input.data == {"a": 1}
    assert "\n" in run_input.text

    txt_path = safe_root / "in.txt"
    txt_path.write_text("hello", encoding="utf-8")
    run_input = io_utils.load_input(txt_path, permissions)
    assert run_input.kind == "text"
    assert run_input.text == "hello"
    assert run_input.data is None


def test_write_output_creates_parent(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    out_path = safe_root / "nested" / "out.txt"
    io_utils.write_output(out_path, "data", permissions)
    assert out_path.read_text(encoding="utf-8") == "data"


def test_make_user_prompt_contains_sections() -> None:
    run_input = RunInput(source_path="file.txt", kind="text", text="hi", data=None)
    prompt = prompting.make_user_prompt("do thing", run_input, {"x": 1})

    assert "# Task Input" in prompt
    assert "source_path: file.txt" in prompt
    assert "## Input Content" in prompt
    assert "hi" in prompt
    assert "## Chain State (JSON)" in prompt
    assert '"x": 1' in prompt
    assert "## Step Instructions" in prompt
    assert "do thing" in prompt


def test_resolve_schema_valid_missing_and_invalid() -> None:
    schema_module = types.ModuleType("schemas.tmp")

    class GoodSchema(BaseModel):
        x: int

    class NotSchema:
        pass

    schema_module.__dict__["GoodSchema"] = GoodSchema
    schema_module.__dict__["NotSchema"] = NotSchema
    sys.modules["schemas.tmp"] = schema_module

    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="y"),
        chain=[ChainStepSpec(id="s1", kind="text", prompt_section="step:one")],
        schemas={"Good": "schemas.tmp:GoodSchema", "Bad": "schemas.tmp:NotSchema"},
    )
    loaded = LoadedAgentFile(spec=spec, body="", step_prompts={})

    try:
        assert resolve_schema(loaded, "Good") is GoodSchema

        with pytest.raises(KeyError):
            resolve_schema(loaded, "Missing")

        with pytest.raises(TypeError):
            resolve_schema(loaded, "Bad")
    finally:
        sys.modules.pop("schemas.tmp", None)


def test_load_agent_file_parses_frontmatter_and_steps(tmp_path: Path) -> None:
    md = """---
name: Test
description: Hello
model:
  base_url: http://localhost
  model_name: test
chain:
  - id: one
    kind: text
    prompt_section: step:one
---

## step:one

body.
"""
    md_path = tmp_path / "agent.md"
    md_path.write_text(md, encoding="utf-8")

    loaded = agent_registry.load_agent_file(md_path)
    assert loaded.spec.name == "Test"
    assert loaded.step_prompts["step:one"] == "body."


def test_load_agent_file_model_defaults_when_missing(tmp_path: Path) -> None:
    md = """---
name: Defaults
model: {}
chain:
  - id: one
    kind: text
    prompt_section: step:one
---

## step:one

body.
"""
    md_path = tmp_path / "agent.md"
    md_path.write_text(md, encoding="utf-8")

    loaded = agent_registry.load_agent_file(md_path)
    assert loaded.spec.model.base_url == "https://api.openai.com/v1"
    assert loaded.spec.model.api_key_env == "OPENAI_API_KEY"
    assert loaded.spec.model.model_name == "gpt-5.2"
    assert loaded.spec.model.provider == "openai-compatible"


def test_load_agent_file_allows_missing_model_block(tmp_path: Path) -> None:
    md = """---
name: Defaults
chain:
  - id: one
    kind: text
    prompt_section: step:one
---

## step:one

body.
"""
    md_path = tmp_path / "agent.md"
    md_path.write_text(md, encoding="utf-8")

    loaded = agent_registry.load_agent_file(md_path)
    assert loaded.spec.model.base_url == "https://api.openai.com/v1"
    assert loaded.spec.model.api_key_env == "OPENAI_API_KEY"
    assert loaded.spec.model.model_name == "gpt-5.2"
    assert loaded.spec.model.provider == "openai-compatible"
