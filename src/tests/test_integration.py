import os
from pathlib import Path

import httpx
import pytest
from quick_agent.orchestrator import Orchestrator
from pydantic import BaseModel


async def _run_agent(orchestrator: Orchestrator, agent_id: str, input_path: Path) -> str:
    result = await orchestrator.run(agent_id, input_path)
    assert isinstance(result, str)
    return result


async def _run_agent_any(orchestrator: Orchestrator, agent_id: str, input_path: Path) -> BaseModel | str:
    return await orchestrator.run(agent_id, input_path)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Missing required env var: {name}")
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _require_ollama(base_url: str) -> None:
    health_url = base_url.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(health_url)
            response.raise_for_status()
    except Exception:
        pytest.skip(f"Ollama is not reachable at {base_url}")


class ContactInfo(BaseModel):
    name: str
    company: str
    email: str
    phone: str
    role: str | None = None


class ContactSummary(BaseModel):
    contact: ContactInfo
    summary: str


class SingleShotStructuredResult(BaseModel):
    ticket_id: str
    priority: str
    action_items: list[str]


def test_orchestrator_runs_agent_end_to_end(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    from quick_agent.orchestrator import Orchestrator

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-5.2"

    output_path = safe_root / "out" / "result.json"
    agent_md = f"""---
name: Test Agent
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
chain:
  - id: one
    kind: text
    prompt_section: step:one
output:
  format: json
  file: {output_path}
---

## step:one

Say ok.
"""
    (agents_dir / "example.md").write_text(agent_md, encoding="utf-8")

    input_path = safe_root / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent, orchestrator, "example", input_path)
    assert output == "ok"


def test_orchestrator_runs_multi_step_contact_extraction(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    output_path = safe_root / "out" / "result.json"
    agent_md = f"""---
name: Contact Extractor
schemas:
  ContactInfo: test_integration:ContactInfo
  ContactSummary: test_integration:ContactSummary
chain:
  - id: extract
    kind: structured
    prompt_section: step:extract
    output_schema: ContactInfo
  - id: summary
    kind: structured
    prompt_section: step:summary
    output_schema: ContactSummary
output:
  format: json
  file: {output_path}
---

## step:extract

Extract the primary business contact from the conversation as JSON that matches the ContactInfo schema.
The \"name\" field should include the person's full name.
If the role is not explicitly stated, set \"role\" to null.

## step:summary

Produce JSON matching ContactSummary. The summary must be a single sentence and include the contact name and company.
Use the extracted JSON from the chain state as the ContactInfo object.
"""
    (agents_dir / "contact.md").write_text(agent_md, encoding="utf-8")

    conversation = (
        "Alex: Thanks for chatting today. The right point of contact is Avery Chen, our "
        "Head of Partnerships at Acme Robotics. You can reach Avery at avery.chen@acmerobotics.com "
        "or call +1-415-555-0199. Let's follow up next week."
    )
    input_path = safe_root / "input.txt"
    input_path.write_text(conversation, encoding="utf-8")

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent_any, orchestrator, "contact", input_path)
    assert isinstance(output, ContactSummary)
    assert output.contact.name == "Avery Chen"
    assert output.contact.company == "Acme Robotics"
    assert output.contact.email == "avery.chen@acmerobotics.com"
    assert output.contact.phone == "+1-415-555-0199"
    assert output.summary
    assert "Avery" in output.summary
    assert "Acme" in output.summary
    assert output_path.exists()
    file_output = ContactSummary.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert file_output.model_dump() == output.model_dump()


def test_orchestrator_runs_true_single_shot_structured_output(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    model_name = os.environ.get("OLLAMA_MODEL") or "gemma3:4b"
    _require_ollama(base_url)

    output_path = safe_root / "out" / "single_shot_result.json"
    agent_md = f"""---
name: Single Shot Structured
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
schemas:
  Output: test_integration:SingleShotStructuredResult
output:
  format: json
  file: {output_path}
  output_schema: Output
---

## Instructions

Extract values from the input and return JSON only for the Output schema.
- ticket_id: copy exactly from input
- priority: copy exactly from input
- action_items: include exactly 2 short items from input
"""
    (agents_dir / "single_shot_structured.md").write_text(agent_md, encoding="utf-8")

    input_path = safe_root / "input.txt"
    input_path.write_text(
        "ticket_id=TCK-219 priority=high action_items=restart service;verify logs",
        encoding="utf-8",
    )

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent_any, orchestrator, "single_shot_structured", input_path)
    assert isinstance(output, SingleShotStructuredResult)
    assert output.ticket_id == "TCK-219"
    assert output.priority.lower() == "high"
    assert len(output.action_items) == 2
    assert output_path.exists()
    file_output = SingleShotStructuredResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert file_output.model_dump() == output.model_dump()


def test_orchestrator_runs_true_single_shot_structured_output_inline_only(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    model_name = os.environ.get("OLLAMA_MODEL") or "gemma3:4b"
    _require_ollama(base_url)

    expected_output_path = safe_root / "out" / "should_not_exist.json"
    agent_md = f"""---
name: Single Shot Structured Inline
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
schemas:
  Output: test_integration:SingleShotStructuredResult
output:
  format: json
  output_schema: Output
---

## Instructions

Extract values from the input and return JSON only for the Output schema.
- ticket_id: copy exactly from input
- priority: copy exactly from input
- action_items: include exactly 2 short items from input
"""
    (agents_dir / "single_shot_structured_inline.md").write_text(agent_md, encoding="utf-8")

    input_path = safe_root / "input.txt"
    input_path.write_text(
        "ticket_id=TCK-220 priority=medium action_items=rotate key;notify owner",
        encoding="utf-8",
    )

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent_any, orchestrator, "single_shot_structured_inline", input_path)
    assert isinstance(output, SingleShotStructuredResult)
    assert output.ticket_id == "TCK-220"
    assert output.priority.lower() == "medium"
    assert len(output.action_items) == 2
    assert not expected_output_path.exists()


def test_orchestrator_allows_agent_call_tool(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    child_output = safe_root / "out" / "child.json"
    child_md = f"""---
name: Child Agent
chain:
  - id: respond
    kind: text
    prompt_section: step:respond
output:
  format: json
  file: {child_output}
---

## step:respond

Reply with exactly: pong
"""
    (agents_dir / "child.md").write_text(child_md, encoding="utf-8")

    parent_output = safe_root / "out" / "parent.json"
    parent_md = f"""---
name: Parent Agent
tools:
  - "agent_call"
nested_output: inline
chain:
  - id: invoke
    kind: text
    prompt_section: step:invoke
output:
  format: json
  file: {parent_output}
---

## step:invoke

Call agent_call with agent "child" and input_file "{{base_directory}}/child_input.txt".
Then respond with only the returned text value.
"""
    (agents_dir / "parent.md").write_text(parent_md, encoding="utf-8")

    child_input = safe_root / "child_input.txt"
    child_input.write_text("ignored", encoding="utf-8")

    parent_input = safe_root / "parent_input.txt"
    parent_input.write_text("call child", encoding="utf-8")

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent, orchestrator, "parent", parent_input)
    assert output == "pong"
    assert not child_output.exists()


def test_orchestrator_allows_agent_call_tool_with_inline_text(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    child_output = safe_root / "out" / "child.json"
    child_md = f"""---
name: Child Agent
chain:
  - id: respond
    kind: text
    prompt_section: step:respond
output:
  format: json
  file: {child_output}
---

## step:respond

Reply with exactly: pong
"""
    (agents_dir / "child.md").write_text(child_md, encoding="utf-8")

    parent_output = safe_root / "out" / "parent.json"
    parent_md = f"""---
name: Parent Agent
tools:
  - "agent_call"
nested_output: inline
chain:
  - id: invoke
    kind: text
    prompt_section: step:invoke
output:
  format: json
  file: {parent_output}
---

## step:invoke

Call agent_call with agent "child" and input_text "hello from memory".
Then respond with only the returned text value.
"""
    (agents_dir / "parent.md").write_text(parent_md, encoding="utf-8")

    parent_input = safe_root / "parent_input.txt"
    parent_input.write_text("call child", encoding="utf-8")

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent, orchestrator, "parent", parent_input)
    assert output == "pong"
    assert not child_output.exists()


def test_orchestrator_allows_nested_output_file(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    child_output = safe_root / "out" / "child.json"
    child_md = f"""---
name: Child Agent
chain:
  - id: respond
    kind: text
    prompt_section: step:respond
output:
  format: json
  file: {child_output}
---

## step:respond

Reply with exactly: pong
"""
    (agents_dir / "child.md").write_text(child_md, encoding="utf-8")

    parent_output = safe_root / "out" / "parent.json"
    parent_md = f"""---
name: Parent Agent
tools:
  - "agent_call"
nested_output: file
chain:
  - id: invoke
    kind: text
    prompt_section: step:invoke
output:
  format: json
  file: {parent_output}
---

## step:invoke

Call agent_call with agent "child" and input_text "hello from memory".
Then respond with only the returned text value.
"""
    (agents_dir / "parent.md").write_text(parent_md, encoding="utf-8")

    parent_input = safe_root / "parent_input.txt"
    parent_input.write_text("call child", encoding="utf-8")

    orchestrator = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )

    import anyio

    output = anyio.run(_run_agent, orchestrator, "parent", parent_input)
    assert output == "pong"
    assert child_output.exists()


def test_file_manager_agent_list_find_read_append(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    # Seed test files
    notes_file = safe_root / "meeting_notes.txt"
    notes_file.write_text("Original meeting notes.\n", encoding="utf-8")
    (safe_root / "report_q4.txt").write_text("Q4 report data.\n", encoding="utf-8")
    (safe_root / "readme.md").write_text("# README\n", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4o"

    agent_md = f"""---
name: File Manager Agent
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
  temperature: 0.1
  max_tokens: 2048
tools:
  - "filesystem_list_files"
  - "filesystem_find_closest_file"
  - "filesystem_read_text"
  - "filesystem_append_text"
chain:
  - id: execute
    kind: text
    prompt_section: step:execute
output:
  format: json
  file: out/result.json
---

## step:execute

You are given a JSON input with keys: directory, search_name, append_text.

Follow these steps in order:
1. Call filesystem_list_files with the given directory to see available files.
2. Call filesystem_find_closest_file with the directory and search_name to get the full path of the closest matching file.
3. Call filesystem_read_text with the full path returned in step 2.
4. Call filesystem_append_text with the same path and the append_text value.
5. Return a plain-text summary including the file found and what was appended.
"""
    (agents_dir / "file-manager.md").write_text(agent_md, encoding="utf-8")

    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent

    input_path = safe_root / "input.json"
    input_path.write_text(
        f'{{"directory": "{safe_root}", "search_name": "meeting", "append_text": "\\nAppended line."}}',
        encoding="utf-8",
    )

    orchestrator = Orchestrator(
        [agents_dir],
        [system_tools_dir],
        safe_dir=safe_root,
    )

    import anyio

    anyio.run(_run_agent, orchestrator, "file-manager", input_path)

    updated = notes_file.read_text(encoding="utf-8")
    assert "Original meeting notes." in updated
    assert "Appended line." in updated
