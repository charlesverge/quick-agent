"""Prompt composition helpers."""

from __future__ import annotations

from typing import Any, Mapping

import yaml

from quick_agent.models.run_input import RunInput


def make_user_prompt(run_input: RunInput, state: Mapping[str, Any]) -> str:
    """
    Creates a consistent user prompt payload. Consistency helps prefix-caching backends.
    """
    # Keep the preamble stable; append variable fields below.
    is_inline = run_input.source_path == "inline_input.txt"
    steps_state = state.get("steps") if isinstance(state, Mapping) else None
    has_state = bool(steps_state)
    include_input_header = not (is_inline and not has_state)

    lines: list[str] = []
    if not is_inline:
        lines.extend(
            [
                "# Task Input",
                f"source_path: {run_input.source_path}",
                f"kind: {run_input.kind}",
                "",
            ]
        )

    if include_input_header:
        lines.append("## Input Content")
    lines.append(run_input.text)

    if has_state:
        state_yaml = yaml.safe_dump(
            steps_state,
            allow_unicode=False,
            default_flow_style=False,
            sort_keys=True,
        ).rstrip()
        lines.extend(["", "## Chain State (YAML)", state_yaml])

    return "\n".join(lines).rstrip() + "\n"
