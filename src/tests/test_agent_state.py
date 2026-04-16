"""Unit tests for AgentState tool detection, memory serialization, and state sync."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from quick_agent.agent_state import AgentState
from quick_agent.models.batch_request import (
    BatchAgentContext,
    BatchModelConfig,
    BatchMessage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_AGENTS = REPO_ROOT / "deploy" / "bedrock-batch-test-harness" / "agents"


# ---------------------------------------------------------------------------
# AgentState dataclass
# ---------------------------------------------------------------------------


def test_agent_state_holds_memory() -> None:
    state = AgentState(memory={"first_name": "Charles"})
    assert state.memory["first_name"] == "Charles"


def test_agent_state_empty_memory() -> None:
    state = AgentState(memory={})
    assert state.memory == {}


# ---------------------------------------------------------------------------
# AgentState parameter detection (via typing.get_type_hints in executor)
# ---------------------------------------------------------------------------


def test_state_tool_invocation() -> None:
    """Directly confirm AgentState tool receives correct memory."""
    state = AgentState(memory={"first_name": "Alice"})
    result = _personalize(state, "hello")
    assert result == "Alice hello"


def _personalize(state: AgentState, word: str) -> str:
    """Helper tool that uses AgentState — no from __future__ annotations effect."""
    return f"{state.memory['first_name']} {word}"


# ---------------------------------------------------------------------------
# BatchAgentContext memory serialization
# ---------------------------------------------------------------------------


def test_batch_agent_context_default_memory() -> None:
    ctx = BatchAgentContext()
    assert ctx.memory == {}


def test_batch_agent_context_with_memory() -> None:
    ctx = BatchAgentContext(memory={"first_name": "Charles"})
    assert ctx.memory["first_name"] == "Charles"


def test_batch_agent_context_round_trips_memory() -> None:
    ctx = BatchAgentContext(memory={"first_name": "Charles", "score": 42})
    dumped = ctx.model_dump(mode="json")
    restored = BatchAgentContext.model_validate(dumped)
    assert restored.memory == {"first_name": "Charles", "score": 42}


# ---------------------------------------------------------------------------
# load_batch_context state sync
# ---------------------------------------------------------------------------


def _make_agent(memory: dict[str, object]) -> Any:
    from quick_agent.agent_registry import AgentRegistry
    from quick_agent.agent_tools import AgentTools
    from quick_agent.directory_permissions import DirectoryPermissions
    from quick_agent.input_adaptors import TextInput
    from quick_agent.quick_agent import QuickAgent

    registry = AgentRegistry([HARNESS_AGENTS])
    tools = AgentTools([REPO_ROOT / "examples" / "agent_memory"])
    permissions = DirectoryPermissions(None)
    return QuickAgent(
        registry=registry,
        tools=tools,
        directory_permissions=permissions,
        agent_id="harness-agent-memory",
        input_data=TextInput("probe"),
        extra_tools=None,
        memory=memory,
    )


def test_load_batch_context_syncs_executor_state() -> None:
    with patch("quick_agent.agent_tools.AgentTools.build_toolset", return_value=None):
        agent = _make_agent({"first_name": "Charles"})
    assert agent._executor.config.state is agent.state

    ctx = BatchAgentContext(
        input_text="probe",
        state={
            "agent_id": "harness-agent-memory",
            "steps": {"generate_random_word": {"random_word": "quartz"}},
            "last_step_output": {"random_word": "quartz"},
        },
        memory={"first_name": "Charles"},
    )
    agent.load_batch_context(context=ctx)

    assert agent.state["steps"] == {"generate_random_word": {"random_word": "quartz"}}
    # After load_batch_context the executor state must point to the new dict.
    assert agent._executor.config.state is agent.state


def test_executor_state_mutation_visible_in_quick_agent() -> None:
    """Mutations to self.state after load_batch_context are shared with executor."""
    with patch("quick_agent.agent_tools.AgentTools.build_toolset", return_value=None):
        agent = _make_agent({})
    ctx = BatchAgentContext(
        input_text="probe",
        state={
            "agent_id": "harness-agent-memory",
            "steps": {},
            "last_step_output": None,
        },
        memory={},
    )
    agent.load_batch_context(context=ctx)
    # Mutate via agent.state directly (as _import_chain_result does)
    agent.state["steps"]["generate_random_word"] = {"random_word": "oak"}
    # Executor config must reflect the mutation
    assert agent._executor.config.state["steps"]["generate_random_word"] == {
        "random_word": "oak"
    }


# ---------------------------------------------------------------------------
# End-to-end: _execute_tool_calls passes AgentState when first param matches
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_tool_calls_injects_agent_state() -> None:
    from quick_agent.executor import AgentExecutor
    from quick_agent.agent_config import AgentConfig
    from quick_agent.models.loaded_agent_file import LoadedAgentFile
    from quick_agent.models.model_spec import ModelSpec
    from quick_agent.models.run_input import RunInput
    from quick_agent.models.batch_request import BatchImportOutcome
    from quick_agent.models.output_spec import OutputSpec
    from quick_agent.models import AgentSpec

    received: list[AgentState] = []

    # This function intentionally has no `from __future__ import annotations`
    # effect — it is defined at call time inside the function body, so Python
    # 3.14 lazily evaluates annotations, but get_type_hints resolves them.
    def my_state_tool(state: AgentState, word: str) -> str:
        received.append(state)
        return f"{state.memory['first_name']} {word}"

    tool_mock = MagicMock()
    tool_mock.function = my_state_tool
    toolset_mock = MagicMock()
    toolset_mock.tools = {"my_state_tool": tool_mock}

    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            provider="openai-compatible", base_url="http://localhost", model_name="m"
        ),
        output=OutputSpec(format="text"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="", kind="text", text="probe")
    config = AgentConfig(
        agent_id="test",
        toolset=toolset_mock,
        tool_ids=["my_state_tool"],
        memory={"first_name": "Charles"},
        model_spec=spec.model,
        client=None,
        http_client=None,
        extra_headers=None,
        extra_body=None,
        record_http_traffic=False,
        run_input=run_input,
        loaded=loaded,
        extra_tools=None,
        recorder=None,
        state={},
    )
    executor = AgentExecutor(config=config)

    tool_calls: list[dict[str, object]] = [
        {"id": "tc-1", "name": "my_state_tool", "arguments": {"word": "quartz"}}
    ]
    results = await executor._execute_tool_calls(tool_calls)

    assert len(results) == 1
    assert results[0].content == "Charles quartz"
    assert results[0].error is None
    assert len(received) == 1
    assert received[0].memory == {"first_name": "Charles"}


@pytest.mark.anyio
async def test_execute_tool_calls_plain_tool_no_state() -> None:
    from quick_agent.executor import AgentExecutor
    from quick_agent.agent_config import AgentConfig
    from quick_agent.models.loaded_agent_file import LoadedAgentFile
    from quick_agent.models.run_input import RunInput
    from quick_agent.models.batch_request import BatchImportOutcome
    from quick_agent.models.output_spec import OutputSpec
    from quick_agent.models import AgentSpec
    from quick_agent.models.model_spec import ModelSpec

    def plain_tool(word: str) -> str:
        return f"echo {word}"

    tool_mock = MagicMock()
    tool_mock.function = plain_tool
    toolset_mock = MagicMock()
    toolset_mock.tools = {"plain_tool": tool_mock}

    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            provider="openai-compatible", base_url="http://localhost", model_name="m"
        ),
        output=OutputSpec(format="text"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec, instructions="", system_prompt="", step_prompts={}
    )
    run_input = RunInput(source_path="", kind="text", text="probe")
    config = AgentConfig(
        agent_id="test",
        toolset=toolset_mock,
        tool_ids=["plain_tool"],
        memory={},
        model_spec=spec.model,
        client=None,
        http_client=None,
        extra_headers=None,
        extra_body=None,
        record_http_traffic=False,
        run_input=run_input,
        loaded=loaded,
        extra_tools=None,
        recorder=None,
        state={},
    )
    executor = AgentExecutor(config=config)

    results = await executor._execute_tool_calls(
        [{"id": "tc-2", "name": "plain_tool", "arguments": {"word": "oak"}}]
    )

    assert len(results) == 1
    assert results[0].content == "echo oak"
    assert results[0].error is None
