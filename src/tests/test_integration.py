from pathlib import Path
import os

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
    return value


class ContactInfo(BaseModel):
    name: str
    company: str
    email: str
    phone: str
    role: str | None = None


class ContactSummary(BaseModel):
    contact: ContactInfo
    summary: str


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
  - "agent.call"
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
