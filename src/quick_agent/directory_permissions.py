"""Directory permission enforcement for file access."""

from __future__ import annotations

from pathlib import Path


class DirectoryPermissions:
    def __init__(self, root: Path | None) -> None:
        self._root = root.expanduser().resolve(strict=False) if root is not None else None

    @property
    def root(self) -> Path | None:
        return self._root

    def scoped(self, directory: str | None) -> "DirectoryPermissions":
        if self._root is None:
            return self
        if directory:
            candidate = (self._root / directory).expanduser().resolve(strict=False)
            root_resolved = self._root.expanduser().resolve(strict=False)
            if not candidate.is_relative_to(root_resolved):
                raise ValueError(f"Scoped directory {directory!r} escapes safe root {root_resolved}.")
            return DirectoryPermissions(candidate)
        return self

    def resolve(self, path: Path, *, for_write: bool) -> Path:
        if self._root is None:
            raise PermissionError("No safe directory configured; reads and writes are denied.")
        target = path
        if not target.is_absolute():
            target = self._root / target
        resolved = target.expanduser().resolve(strict=False)
        root_resolved = self._root.expanduser().resolve(strict=False)
        if not resolved.is_relative_to(root_resolved):
            raise PermissionError(f"Path {resolved} is outside safe directory {root_resolved}.")
        if for_write:
            root_resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def can_read(self, path: Path) -> bool:
        try:
            self.resolve(path, for_write=False)
        except PermissionError:
            return False
        return True

    def can_write(self, path: Path) -> bool:
        try:
            self.resolve(path, for_write=False)
        except PermissionError:
            return False
        return True
