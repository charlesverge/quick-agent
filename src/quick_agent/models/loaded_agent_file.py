"""Loaded agent markdown plus parsed metadata."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from quick_agent.models.agent_spec import AgentSpec

logger = logging.getLogger(__name__)

SECTION_HEADER_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
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
        safe_dir = str(Path(agent).parent)
        sections = parse_agent_sections(post.content, safe_dir=safe_dir)
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


def parse_fence_marker(line: str) -> tuple[str, int] | None:
    stripped = line.lstrip()
    if not stripped:
        return None
    marker_char = stripped[0]
    if marker_char not in ("`", "~"):
        return None
    marker_len = 0
    for char in stripped:
        if char == marker_char:
            marker_len += 1
        else:
            break
    if marker_len < 3:
        return None
    return (marker_char, marker_len)


def parse_agent_sections(markdown_body: str, safe_dir: str) -> ParsedAgentSections:
    recognized: list[tuple[str, str, int, int]] = []
    instructions_start: int | None = None
    system_prompt_start: int | None = None
    in_fence = False
    active_fence_char = ""
    active_fence_len = 0
    offset = 0

    markdown_body = resolve_includes(markdown_body, safe_dir)

    for line in markdown_body.splitlines(keepends=True):
        fence_marker = parse_fence_marker(line)
        if in_fence:
            if fence_marker is not None:
                marker_char, marker_len = fence_marker
                if marker_char == active_fence_char and marker_len >= active_fence_len:
                    in_fence = False
                    active_fence_char = ""
                    active_fence_len = 0
            offset += len(line)
            continue
        if fence_marker is not None:
            marker_char, marker_len = fence_marker
            in_fence = True
            active_fence_char = marker_char
            active_fence_len = marker_len
            offset += len(line)
            continue

        line_text = line.rstrip("\r\n")
        match = SECTION_HEADER_LINE_RE.match(line_text)
        if match is not None:
            header_text = match.group(2).strip()
            classified = classify_section_header(header_text)
            if classified is not None:
                kind, key = classified
                line_start = offset
                line_end = offset + len(line_text)
                recognized.append((kind, key, line_start, line_end))
                if kind == "instructions" and instructions_start is None:
                    instructions_start = line_start
                if kind == "system_prompt" and system_prompt_start is None:
                    system_prompt_start = line_start
        offset += len(line)

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

def resolve_includes(filename, safe_dir, seen=None):
    if seen is None:
        seen = set()

    # 1. Normalize and resolve paths (resolves symlinks and ../)
    abs_safe_dir = os.path.realpath(safe_dir)

    if os.path.exists(filename):
        abs_target_path = os.path.realpath(filename)

        # 2. SAFE DIR CHECK: Ensure the file is inside the safe directory
        if not abs_target_path.startswith(abs_safe_dir):
            return f"<!-- Access Denied: {filename} is outside safe directory -->"

        # 3. INFINITY CHECK: Prevent circular references
        if abs_target_path in seen:
            return f"<!-- Circular reference detected: {filename} -->"

        if not os.path.exists(abs_target_path):
            return f"<!-- File not found: {filename} -->"

        # Track this file in the current branch
        seen.add(abs_target_path)

        with open(abs_target_path, 'r') as f:
            content = f.read()

        current_file_dir = os.path.dirname(abs_target_path)
    else:
        looks_like_markdown = (
            "\n" in filename
            or "\r" in filename
            or filename.lstrip().startswith("#")
            or "{!" in filename
        )
        if not looks_like_markdown:
            return f"<!-- File not found: {filename} -->"

        content = filename
        current_file_dir = abs_safe_dir

    pattern = re.compile(r'\{!\s*(.*?)\s*!\}')

    def replace_match(match):
        include_path = match.group(1)
        # Resolve path relative to the file containing the include
        full_include_path = os.path.join(current_file_dir, include_path)

        # Pass a copy of 'seen' so siblings can share sub-includes
        return resolve_includes(full_include_path, abs_safe_dir, seen.copy())

    return pattern.sub(replace_match, content)
