"""Agent registry and parsing utilities."""

from __future__ import annotations

from pathlib import Path

from quick_agent.models.loaded_agent_file import LoadedAgentFile, parse_agent_sections


def split_step_sections(markdown_body: str) -> dict[str, str]:
    """
    Extracts blocks that begin with headings like "# step:<id>".
    Returns mapping: "step:<id>" -> content for that step (excluding heading line).
    """
    sections = parse_agent_sections(markdown_body)
    return sections.step_prompts


class AgentRegistry:
    def __init__(self, agent_roots: list[Path]) -> None:
        self.agent_roots: list[Path] = agent_roots
        self._cache: dict[str, LoadedAgentFile] = {}
        self._index: dict[str, Path] | None = None

    def _build_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for root in self.agent_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.md"):
                agent_id = path.stem
                if agent_id in index:
                    continue
                index[agent_id] = path
        return index

    def _get_index(self) -> dict[str, Path]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    def list_agents(self) -> list[str]:
        index = self._get_index()
        return sorted(index.keys())

    def get(self, agent_id: str) -> LoadedAgentFile:
        if agent_id in self._cache:
            return self._cache[agent_id]
        index = self._get_index()
        path = index.get(agent_id)
        if path is None:
            raise FileNotFoundError(f"Agent not found: {agent_id} (searched: {self.agent_roots})")
        loaded = LoadedAgentFile(path)
        self._cache[agent_id] = loaded
        return loaded
