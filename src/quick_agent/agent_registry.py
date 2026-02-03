"""Agent registry and parsing utilities."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from quick_agent.models.agent_spec import AgentSpec
from quick_agent.models.loaded_agent_file import LoadedAgentFile


def split_step_sections(markdown_body: str) -> dict[str, str]:
    """
    Extracts blocks that begin with headings "## step:<id>".
    Returns mapping: "step:<id>" -> content for that step (excluding heading line).
    """
    pattern = re.compile(r"^##\s+(step:[A-Za-z0-9_\-]+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown_body))
    out: dict[str, str] = {}

    for i, m in enumerate(matches):
        section_name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(markdown_body)
        out[section_name] = markdown_body[start:end].strip()

    return out


def load_agent_file(path: Path) -> LoadedAgentFile:
    post = frontmatter.load(str(path))
    spec = AgentSpec.model_validate(post.metadata)
    steps = split_step_sections(post.content)
    return LoadedAgentFile(spec=spec, body=post.content, step_prompts=steps)


class AgentRegistry:
    def __init__(self, agent_roots: list[Path]):
        self.agent_roots = agent_roots
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
        loaded = load_agent_file(path)
        self._cache[agent_id] = loaded
        return loaded
