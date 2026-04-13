from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.io_utils import write_output as write_text
from quick_agent.types import AgentResult


def write_output(
    output_file: Path | str,
    last_step_output: AgentResult,
    permissions: DirectoryPermissions,
) -> Path:
    if not output_file:
        raise ValueError("Output file is not configured.")

    out_path = Path(output_file)
    if isinstance(last_step_output, BaseModel):
        text = last_step_output.model_dump_json(indent=2)
    elif isinstance(last_step_output, (dict, list)):
        text = json.dumps(last_step_output, indent=2)
    else:
        text = str(last_step_output)

    write_text(out_path, text, permissions)
    return out_path
