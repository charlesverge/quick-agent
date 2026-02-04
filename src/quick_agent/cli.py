"""CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from quick_agent.input_adaptors import InputAdaptor, TextInput
from quick_agent.orchestrator import Orchestrator


async def run_agent(
    orch: Orchestrator,
    agent_id: str,
    input_adaptor: InputAdaptor | Path,
    extra_tools: list[str],
) -> BaseModel | str:
    return await orch.run(agent_id, input_adaptor, extra_tools=extra_tools)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-dir", type=str, default="agents")
    parser.add_argument("--tools-dir", type=str, default="tools")
    parser.add_argument("--safe-dir", type=str, default="safe")
    parser.add_argument("--agent", type=str, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Path to an input file")
    input_group.add_argument("--input-text", type=str, help="Raw input text")
    parser.add_argument("--tool", action="append", default=[], help="Extra tool IDs to add at runtime")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    system_agents_dir = package_root / "agents"
    system_tools_dir = package_root / "tools"
    user_agents_dir = Path(args.agents_dir)
    user_tools_dir = Path(args.tools_dir)

    agent_roots = [user_agents_dir, system_agents_dir]
    tool_roots = [user_tools_dir, system_tools_dir]

    orch = Orchestrator(agent_roots, tool_roots, Path(args.safe_dir))
    input_adaptor: InputAdaptor | Path
    if args.input_text is not None:
        input_adaptor = TextInput(args.input_text)
    else:
        input_adaptor = Path(args.input)

    # Async entrypoint
    import anyio

    out = anyio.run(run_agent, orch, args.agent, input_adaptor, args.tool)
    if isinstance(out, BaseModel):
        print(out.model_dump_json(indent=2))
    else:
        print(out)
