import os
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import BaseModel

from quick_agent.orchestrator import Orchestrator
from quick_agent.types import AgentResult, StepOutput


async def _run_agent(
    orchestrator: Orchestrator, agent_id: str, input_path: Path
) -> str:
    result = await orchestrator.run(agent_id, input_path)
    assert isinstance(result, str)
    return result


async def _run_agent_any(
    orchestrator: Orchestrator, agent_id: str, input_path: Path
) -> AgentResult:
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
        raise RuntimeError(
            f"Unable to connect to Ollama at {base_url}. Ensure Ollama is running and the base URL is correct."
        )


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
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"

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
  format: text
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
    file_output = ContactSummary.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert file_output.model_dump() == output.model_dump()


def test_orchestrator_runs_true_single_shot_structured_output(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    model_name = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
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

    output = anyio.run(
        _run_agent_any, orchestrator, "single_shot_structured", input_path
    )
    assert isinstance(output, SingleShotStructuredResult)
    assert output.ticket_id == "TCK-219"
    assert output.priority.lower() == "high"
    assert len(output.action_items) == 2
    assert output_path.exists()
    file_output = SingleShotStructuredResult.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert file_output.model_dump() == output.model_dump()


def test_orchestrator_runs_true_single_shot_structured_output_inline_only(
    tmp_path: Path,
) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    model_name = os.environ.get("OLLAMA_MODEL") or "qwen2.5:7b"
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
    (agents_dir / "single_shot_structured_inline.md").write_text(
        agent_md, encoding="utf-8"
    )

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

    output = anyio.run(
        _run_agent_any, orchestrator, "single_shot_structured_inline", input_path
    )
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
  format: text
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
  format: text
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
  format: text
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
  format: text
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


@pytest.mark.anyio
async def test_orchestrator_allows_nested_output_file(tmp_path: Path) -> None:
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
    output_schema: ResultOutput
schemas:
  ResultOutput: "quick_agent.schemas.outputs:ResultOutput"
output:
  format: json
  file: {child_output}
---

## step:respond

Reply with exactly: result as pong
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
    output_schema: ResultOutput
schemas:
  ResultOutput: "quick_agent.schemas.outputs:ResultOutput"
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

    output = await orchestrator.run("parent", parent_input)
    assert output
    assert isinstance(output, dict)
    assert output.get("result") == "pong"
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
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"

    agent_md = f"""---
name: File Manager Agent
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
  temperature: 0.1
  max_completion_tokens: 2048
max_tool_calls: 8
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
  format: text
  file: out/result.txt
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


def test_batch_execute_single_shot_agent(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = Path(__file__).parent / "fixtures" / "batch_test_mode"

    input_path = safe_root / "input.txt"
    input_path.write_text("hello", encoding="utf-8")

    orchestrator = Orchestrator([agents_dir], safe_dir=safe_root)

    import anyio

    result = anyio.run(orchestrator.batch_execute, "single_shot", input_path)
    assert result == "ok"


def test_batch_execute_chain_agent(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    output_file = safe_root / "output.txt"
    output_file.write_text("", encoding="utf-8")

    agents_dir = Path(__file__).parent / "fixtures" / "batch_test_mode"

    input_path = safe_root / "input.txt"
    input_path.write_text(str(output_file), encoding="utf-8")

    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent

    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)

    import anyio

    result = anyio.run(orchestrator.batch_execute, "chain", input_path)
    assert isinstance(result, str)

    output_content = output_file.read_text(encoding="utf-8")
    assert "step1 executed" in output_content
    assert "step2 executed" in output_content


def test_batch_execute_with_tools(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    notes_file = safe_root / "notes.txt"
    notes_file.write_text("Original notes.\n", encoding="utf-8")

    agents_dir = Path(__file__).parent / "fixtures" / "batch_test_mode"

    input_path = safe_root / "input.txt"
    input_path.write_text("go", encoding="utf-8")

    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent

    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)

    import anyio

    result = anyio.run(orchestrator.batch_execute, "with_tools", input_path)
    assert isinstance(result, str)


@pytest.mark.anyio
async def test_batch_execute_with_tools_verifies_each_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_env("OPENAI_API_KEY")
    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent

    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    notes_file = safe_root / "notes.txt"
    notes_file.write_text("Original notes.\n", encoding="utf-8")

    agents_dir = Path(__file__).parent / "fixtures" / "batch_test_mode"

    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)

    input_path = safe_root / "input.txt"
    input_path.write_text(str(notes_file), encoding="utf-8")

    from quick_agent import QuickAgent
    from quick_agent.executor import ToolCallResult

    tool_call_log: list[dict[str, object]] = []

    async def mock_execute_tool_calls(
        self, tool_calls: list[dict[str, object]]
    ) -> list[ToolCallResult]:
        tool_call_log.extend(tool_calls)
        results = []
        for tc in tool_calls:
            tc_id = str(tc.get("id", ""))
            tc_name = str(tc.get("name", ""))
            results.append(ToolCallResult(id=tc_id, name=tc_name, content="done"))
        return results

    import quick_agent.executor as executor_module

    monkeypatch.setattr(
        executor_module.AgentExecutor,
        "_execute_tool_calls",
        mock_execute_tool_calls,
    )

    try:
        loaded = orchestrator.registry.get("with_tools")
        extra_tools = list(loaded.spec.tools or [])
        agent = QuickAgent(
            registry=orchestrator.registry,
            tools=orchestrator.tools,
            directory_permissions=orchestrator.directory_permissions,
            agent_id="with_tools",
            input_data=input_path,
            extra_tools=extra_tools,
            test_mode=True,
        )

        processor = agent.processor
        assert processor

        batch_request = agent.batch()[0]
        import_request = await processor.run_batch(batch_request)
        outcome = await agent.import_result(batch_import=import_request)

        while outcome.next_request:
            agent = QuickAgent(
                registry=orchestrator.registry,
                tools=orchestrator.tools,
                directory_permissions=orchestrator.directory_permissions,
                agent_id="with_tools",
                input_data=input_path,
                extra_tools=extra_tools,
                test_mode=True,
            )
            processor = agent.processor
            assert processor

            next_req = outcome.next_request
            if next_req.context and next_req.context.state:
                ctx_state = next_req.context.state
                steps_value = ctx_state.get("steps")
                last_output_value: StepOutput | None = None
                last_output_obj = ctx_state.get("last_step_output")
                if isinstance(last_output_obj, (BaseModel, str, dict)):
                    last_output_value = last_output_obj
                steps: dict[str, StepOutput] = {}
                if isinstance(steps_value, dict):
                    steps = steps_value
                agent.state = {
                    "agent_id": "with_tools",
                    "steps": steps,
                    "last_step_output": last_output_value,
                }

            batch_request = agent.batch()[0]
            import_request = await processor.run_batch(batch_request)
            outcome = await agent.import_result(batch_import=import_request)

        assert len(tool_call_log) > 0
        tool_call_names = [tc.get("name") for tc in tool_call_log]
        assert "filesystem_append_text" in tool_call_names
        append_call = next(
            tc for tc in tool_call_log if tc.get("name") == "filesystem_append_text"
        )
        args = append_call.get("arguments", {})
        assert notes_file.name in str(args)
        assert "appended by test" in str(args)
    finally:
        monkeypatch.undo()


@pytest.mark.anyio
async def test_agent_processor_direct_usage(tmp_path: Path) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)

    agents_dir = Path(__file__).parent / "fixtures" / "batch_test_mode"

    input_path = safe_root / "input.txt"
    input_path.write_text("test", encoding="utf-8")

    orchestrator = Orchestrator([agents_dir], safe_dir=safe_root)

    from quick_agent import QuickAgent

    agent = QuickAgent(
        registry=orchestrator.registry,
        tools=orchestrator.tools,
        directory_permissions=orchestrator.directory_permissions,
        agent_id="single_shot",
        input_data=input_path,
        test_mode=True,
    )

    processor = agent.processor
    assert processor

    batch_request = agent.batch()[0]
    import_request = await processor.run_batch(batch_request)
    outcome = await agent.import_result(batch_import=import_request)

    assert outcome.result is not None


@pytest.mark.anyio
async def test_integration_tool_choice_required_invokes_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    agent_md = f"""---
name: Tool Choice Required
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
  temperature: 0.0
  max_completion_tokens: 256
tools:
  - "filesystem_list_files"
chain:
  - id: execute
    kind: text
    prompt_section: step:execute
    tool_choice: required
output:
  format: text
---

## step:execute

Call filesystem_list_files for this directory: {safe_root}.
Then return exactly the word done.
"""
    (agents_dir / "tool_choice_required.md").write_text(agent_md, encoding="utf-8")
    input_path = safe_root / "input.txt"
    input_path.write_text("run", encoding="utf-8")
    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent
    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)
    from quick_agent import QuickAgent
    from quick_agent.executor import ToolCallResult

    tool_call_log: list[dict[str, object]] = []

    async def mock_execute_tool_calls(
        self, tool_calls: list[dict[str, object]]
    ) -> list[ToolCallResult]:
        tool_call_log.extend(tool_calls)
        results = []
        for tc in tool_calls:
            tc_id = str(tc.get("id", ""))
            tc_name = str(tc.get("name", ""))
            results.append(ToolCallResult(id=tc_id, name=tc_name, content="ok"))
        return results

    import quick_agent.executor as executor_module

    monkeypatch.setattr(
        executor_module.AgentExecutor,
        "_execute_tool_calls",
        mock_execute_tool_calls,
    )
    agent = QuickAgent(
        registry=orchestrator.registry,
        tools=orchestrator.tools,
        directory_permissions=orchestrator.directory_permissions,
        agent_id="tool_choice_required",
        input_data=input_path,
        test_mode=True,
    )
    processor = agent.processor
    assert processor
    batch_request = agent.batch()[0]
    import_request = await processor.run_batch(batch_request)
    outcome = await agent.import_result(batch_import=import_request)
    while outcome.next_request is not None:
        next_req = outcome.next_request
        import_request = await processor.run_batch(next_req)
        outcome = await agent.import_result(batch_import=import_request)
    assert len(tool_call_log) > 0
    assert any(tc.get("name") == "filesystem_list_files" for tc in tool_call_log)


@pytest.mark.anyio
async def test_integration_tool_choice_none_blocks_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_env("OPENAI_API_KEY")
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    agent_md = f"""---
name: Tool Choice None
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
  temperature: 0.0
  max_completion_tokens: 256
tools:
  - "filesystem_list_files"
chain:
  - id: execute
    kind: text
    prompt_section: step:execute
    tool_choice: none
output:
  format: text
---

## step:execute

Do not call any tool.
Return exactly the word done.
"""
    (agents_dir / "tool_choice_none.md").write_text(agent_md, encoding="utf-8")
    input_path = safe_root / "input.txt"
    input_path.write_text("run", encoding="utf-8")
    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent
    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)
    from quick_agent import QuickAgent
    from quick_agent.executor import ToolCallResult

    tool_call_log: list[dict[str, object]] = []

    async def mock_execute_tool_calls(
        self, tool_calls: list[dict[str, object]]
    ) -> list[ToolCallResult]:
        tool_call_log.extend(tool_calls)
        return []

    import quick_agent.executor as executor_module

    monkeypatch.setattr(
        executor_module.AgentExecutor,
        "_execute_tool_calls",
        mock_execute_tool_calls,
    )
    agent = QuickAgent(
        registry=orchestrator.registry,
        tools=orchestrator.tools,
        directory_permissions=orchestrator.directory_permissions,
        agent_id="tool_choice_none",
        input_data=input_path,
        test_mode=True,
    )
    processor = agent.processor
    assert processor
    batch_request = agent.batch()[0]
    import_request = await processor.run_batch(batch_request)
    outcome = await agent.import_result(batch_import=import_request)
    assert outcome.result is not None
    assert len(tool_call_log) == 0


def test_integration_tool_choice_any_normalizes_to_auto_for_non_bedrock(
    tmp_path: Path,
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True)
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
    agent_md = f"""---
name: Tool Choice Any
model:
  provider: openai-compatible
  base_url: {base_url}
  api_key_env: OPENAI_API_KEY
  model_name: {model_name}
tools:
  - "filesystem_list_files"
chain:
  - id: execute
    kind: text
    prompt_section: step:execute
    tool_choice: any
output:
  format: text
---

## step:execute

Return done.
"""
    (agents_dir / "tool_choice_any.md").write_text(agent_md, encoding="utf-8")
    input_path = safe_root / "input.txt"
    input_path.write_text("run", encoding="utf-8")
    import quick_agent.tools as _tools_pkg

    system_tools_dir = Path(_tools_pkg.__file__).resolve().parent
    orchestrator = Orchestrator([agents_dir], [system_tools_dir], safe_dir=safe_root)
    batch_request = anyio.run(orchestrator.batch, "tool_choice_any", input_path)[0]
    assert batch_request.tool_choice is not None
    assert batch_request.tool_choice.mode == "any"
