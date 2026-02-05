# Agents
test

<INSTRUCTIONS>
## Project Rules
- When executing python scripts use ./.venv/bin/python
- Do not use `typing.cast` to `Any` (or `cast(Any, ...)`) in this codebase.
- Do not use `eval` or `exec` in this codebase. ,
- Do not use TYPE_CHECKING imports with fallback TypedDicts (no `if TYPE_CHECKING: ... else: class X(TypedDict, ...)` pattern).
- Do not define nested functions inside other functions or methods unless explicitly asked for.
- Do not use `sys.path.insert` unless explicitly asked for.
- Do not use `# type: ignore` unless explicitly asked for.
- Do not re-export symbols; import from the correct module instead. Re-exports from `__init__.py` are allowed.
- When edit .md files a new line should follow headers (e.g., `## Header`).
- Documentation must be derived from this project’s source files or included docs (no generic filler).
- When logic is reused across multiple locations, define a single helper and use it everywhere. Avoid duplicating the same inline condition in more than one place.
- Follow user instructions precisely and prefer explicit wording from the user when choosing names, behavior, or structure.
- When creating or modifying markdown ensure it is valid. ie headers have new lines after them
</INSTRUCTIONS>
