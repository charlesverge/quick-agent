<INSTRUCTIONS>
## Project Rules
- Do not use `typing.cast` to `Any` (or `cast(Any, ...)`) in this codebase.
- Do not use `eval` or `exec` in this codebase. ,
- Do not use TYPE_CHECKING imports with fallback TypedDicts (no `if TYPE_CHECKING: ... else: class X(TypedDict, ...)` pattern).
- Do not define nested functions inside other functions or methods unless explicitly asked for.
- Do not use `sys.path.insert` unless explicitly asked for.
- Do not use `# type: ignore` unless explicitly asked for.
- Do not re-export symbols; import from the correct module instead.
</INSTRUCTIONS>
