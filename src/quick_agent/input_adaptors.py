"""Input adaptors for agents."""

from __future__ import annotations

from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.io_utils import load_input
from quick_agent.models.run_input import RunInput


class InputAdaptor:
    def load(self) -> RunInput:
        raise NotImplementedError("InputAdaptor.load must be implemented by subclasses.")


class FileInput(InputAdaptor):
    def __init__(self, path: Path, permissions: DirectoryPermissions) -> None:
        self._run_input = load_input(path, permissions)

    def load(self) -> RunInput:
        return self._run_input


class TextInput(InputAdaptor):
    def __init__(self, text: str) -> None:
        self._text = text

    def load(self) -> RunInput:
        return RunInput(source_path="inline_input.txt", kind="text", text=self._text, data=None)
