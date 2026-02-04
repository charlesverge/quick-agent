"""Prompt composition helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping

from quick_agent.models.run_input import RunInput


def make_user_prompt(step_prompt: str, run_input: RunInput, state: Mapping[str, Any]) -> str:
    """
    Creates a consistent user prompt payload. Consistency helps prefix-caching backends.
    """
    # Keep the preamble stable; append variable fields below.
    return f"""# Task Input
source_path: {run_input.source_path}
kind: {run_input.kind}

## Input Content
{run_input.text}

## Chain State (JSON)
{json.dumps(state, indent=2)}

## Step Instructions
{step_prompt}
"""
