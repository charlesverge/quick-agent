"""Loaded agent markdown plus parsed metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from quick_agent.models.agent_spec import AgentSpec


logger = logging.getLogger(__name__)

SECTION_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
STEP_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class LoadedAgentFile:
    spec: AgentSpec
    instructions: str
    system_prompt: str
    step_prompts: dict[str, str]  # prompt_section -> markdown chunk

    def __init__(self, agent: Path | str) -> None:
        post, source_label = load_agent_frontmatter(agent)
        spec = AgentSpec.model_validate(post.metadata)
        sections = parse_agent_sections(post.content)
        if sections.first_section_start is not None and (
            sections.instructions_start is not None or sections.system_prompt_start is not None
        ):
            preamble = post.content[: sections.first_section_start]
            if preamble.strip():
                logger.warning("Ignored text before instructions or system prompt in %s", source_label)
        if not sections.step_prompts and sections.instructions_start is None and sections.system_prompt_start is None:
            raise ValueError("Agent markdown must include instructions, system prompt, or step sections.")
        self.spec = spec
        self.instructions = sections.instructions
        self.system_prompt = sections.system_prompt
        self.step_prompts = sections.step_prompts

    @classmethod
    def from_parts(
        cls,
        *,
        spec: AgentSpec,
        instructions: str,
        system_prompt: str,
        step_prompts: dict[str, str],
    ) -> "LoadedAgentFile":
        obj = cls.__new__(cls)
        obj.spec = spec
        obj.instructions = instructions
        obj.system_prompt = system_prompt
        obj.step_prompts = step_prompts
        return obj


@dataclass(frozen=True)
class ParsedAgentSections:
    instructions: str
    system_prompt: str
    step_prompts: dict[str, str]
    instructions_start: int | None
    system_prompt_start: int | None
    first_section_start: int | None


def load_agent_frontmatter(agent: Path | str) -> tuple[frontmatter.Post, str]:
    if isinstance(agent, Path):
        post = frontmatter.load(str(agent))
        return post, str(agent)
    agent_path = Path(agent)
    if agent_path.exists():
        post = frontmatter.load(str(agent_path))
        return post, str(agent_path)
    post = frontmatter.loads(agent)
    return post, "<inline>"


def normalize_header_text(header_text: str) -> str:
    normalized = header_text.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", normalized)


def classify_section_header(header_text: str) -> tuple[str, str] | None:
    header = header_text.strip()
    if ":" in header:
        prefix, step_id = header.split(":", 1)
        if prefix.strip().lower() == "step":
            step_id = step_id.strip()
            if step_id and STEP_ID_RE.match(step_id):
                return ("step", f"step:{step_id}")
    normalized = normalize_header_text(header_text)
    if normalized == "instructions":
        return ("instructions", "instructions")
    if normalized == "system prompt":
        return ("system_prompt", "system_prompt")
    return None


def parse_agent_sections(markdown_body: str) -> ParsedAgentSections:
    matches = list(SECTION_HEADER_RE.finditer(markdown_body))
    recognized: list[tuple[str, str, int, int]] = []
    instructions_start: int | None = None
    system_prompt_start: int | None = None

    for match in matches:
        header_text = match.group(2).strip()
        classified = classify_section_header(header_text)
        if classified is None:
            continue
        kind, key = classified
        recognized.append((kind, key, match.start(), match.end()))
        if kind == "instructions" and instructions_start is None:
            instructions_start = match.start()
        if kind == "system_prompt" and system_prompt_start is None:
            system_prompt_start = match.start()

    instructions = ""
    system_prompt = ""
    step_prompts: dict[str, str] = {}

    for index, (kind, key, _start, end) in enumerate(recognized):
        section_start = end
        next_index = index + 1
        section_end = recognized[next_index][2] if next_index < len(recognized) else len(markdown_body)
        content = markdown_body[section_start:section_end].strip()
        if kind == "instructions":
            if not instructions:
                instructions = content
        elif kind == "system_prompt":
            if not system_prompt:
                system_prompt = content
        else:
            step_prompts[key] = content

    first_section_start = recognized[0][2] if recognized else None
    return ParsedAgentSections(
        instructions=instructions,
        system_prompt=system_prompt,
        step_prompts=step_prompts,
        instructions_start=instructions_start,
        system_prompt_start=system_prompt_start,
        first_section_start=first_section_start,
    )
