"""CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from quick_agent.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-dir", type=str, default="agents")
    parser.add_argument("--tools-dir", type=str, default="tools")
    parser.add_argument("--safe-dir", type=str, default="safe")
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
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

    # Async entrypoint
    import anyio

    async def runner():
        return await orch.run(args.agent, Path(args.input), extra_tools=args.tool)

    out = anyio.run(runner)
    if isinstance(out, BaseModel):
        print(out.model_dump_json(indent=2))
    else:
        print(out)
