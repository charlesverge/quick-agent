from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from quick_agent.models.loaded_agent_file import (
    LoadedAgentFile,
    parse_agent_sections,
    resolve_includes,
)


def test_resolve_includes_blocks_path_outside_safe_dir(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    main = safe_dir / "main.md"
    main.write_text("{! ../outside.md !}", encoding="utf-8")
    result = resolve_includes(str(main), str(safe_dir))
    assert "Access Denied" in result
    assert "secret" not in result


def test_resolve_includes_blocks_symlink_escape(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = safe_dir / "link.md"
    os.symlink(outside, link)
    result = resolve_includes(str(link), str(safe_dir))
    assert "Access Denied" in result
    assert "secret" not in result


def test_resolve_includes_detects_circular_reference(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    a = safe_dir / "a.md"
    b = safe_dir / "b.md"
    a.write_text("a content {! b.md !}", encoding="utf-8")
    b.write_text("b content {! a.md !}", encoding="utf-8")
    result = resolve_includes(str(a), str(safe_dir))
    assert "Circular reference detected" in result


def test_resolve_includes_handles_missing_file(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    result = resolve_includes("{! does_not_exist.md !}", str(safe_dir))
    assert "File not found" in result


def test_resolve_includes_allows_sibling_includes(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    shared = safe_dir / "shared.md"
    shared.write_text("shared", encoding="utf-8")
    main = safe_dir / "main.md"
    main.write_text("{! shared.md !} and {! shared.md !}", encoding="utf-8")
    result = resolve_includes(str(main), str(safe_dir))
    assert result.count("shared") == 2
    assert "Circular reference detected" not in result


def test_resolve_includes_nested_three_levels(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    a = safe_dir / "a.md"
    b = safe_dir / "b.md"
    c = safe_dir / "c.md"
    c.write_text("deep", encoding="utf-8")
    b.write_text("b content {! c.md !}", encoding="utf-8")
    a.write_text("a content {! b.md !}", encoding="utf-8")
    result = resolve_includes(str(a), str(safe_dir))
    assert "deep" in result


def test_resolve_includes_inline_markdown_string_no_file_match() -> None:
    source = "# Heading\nSome content"
    result = resolve_includes(source, ".")
    assert result == source


def test_resolve_includes_does_not_treat_directory_name_as_file(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    result = resolve_includes(str(safe_dir), str(safe_dir))
    assert "File not found" in result


def test_resolve_includes_handles_whitespace_around_path(tmp_path: Path) -> None:
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    x = safe_dir / "x.md"
    x.write_text("hello", encoding="utf-8")
    result = resolve_includes("{!  x.md  !}", str(safe_dir))
    assert "hello" in result


def test_parse_agent_sections_ignores_section_headers_inside_code_fences() -> None:
    markdown = "## Instructions\nReal instructions\n```\n## step:fake\nThis should not be parsed\n```\n## step:real\nReal step\n"
    sections = parse_agent_sections(markdown, safe_dir=".")
    assert "step:real" in sections.step_prompts
    assert "step:fake" not in sections.step_prompts
    assert "Real instructions" in sections.instructions
    assert "## step:fake" in sections.instructions


def test_parse_agent_sections_handles_tilde_fences() -> None:
    markdown = "## Instructions\nReal instructions\n~~~\n## step:fake\nThis should not be parsed\n~~~\n## step:real\nReal step\n"
    sections = parse_agent_sections(markdown, safe_dir=".")
    assert "step:real" in sections.step_prompts
    assert "step:fake" not in sections.step_prompts


def test_parse_agent_sections_classifies_step_header_case_sensitively() -> None:
    markdown = "## STEP:foo\nContent\n"
    sections = parse_agent_sections(markdown, safe_dir=".")
    assert "step:foo" in sections.step_prompts


def test_parse_agent_sections_rejects_invalid_step_ids() -> None:
    markdown = "## step:has spaces\nContent\n## step:has.dots\nContent\n"
    sections = parse_agent_sections(markdown, safe_dir=".")
    assert not sections.step_prompts


def test_parse_agent_sections_warns_on_preamble_before_first_section(caplog: pytest.LogCaptureFixture) -> None:
    markdown = "---\nname: test\n---\nSome preamble text\n\n## Instructions\nReal\n"
    with caplog.at_level(logging.WARNING):
        LoadedAgentFile(markdown)
    assert "Ignored text before instructions or system prompt" in caplog.text


def test_parse_agent_sections_first_instructions_wins() -> None:
    markdown = "## Instructions\nFirst\n## Instructions\nSecond\n"
    sections = parse_agent_sections(markdown, safe_dir=".")
    assert sections.instructions == "First"
