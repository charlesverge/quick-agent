# Command Line Usage

This project ships a CLI wrapper in `agent.py` and a module entrypoint in `quick_agent.cli`.

## Quick Start

```bash
python agent.py --agent example --input path/to/input.txt
```

If you want to run the module directly:

```bash
python -m quick_agent.cli --agent example --input path/to/input.txt
```

## CLI Options

- `--agent` (required): Agent id (filename without `.md`).
- `--input`: Input file path (mutually exclusive with `--input-text`). Supports `.txt`, `.md`, and `.json`.
- `--input-text`: Raw input text (mutually exclusive with `--input`).
- `--agents-dir`: Directory to search for user agents. Default: `agents`.
- `--tools-dir`: Directory to search for user tools. Default: `tools`.
- `--safe-dir`: Root directory for file access. Default: `safe`.
- `--tool`: Extra tool ids to add at runtime. Can be repeated.

## Safe Directory

File reads and writes for agent input/output and filesystem tools are restricted to a safe directory.

- Configure with `--safe-dir` (relative to the current working directory if not absolute).
- Paths outside the safe directory raise `PermissionError`.
- Relative paths are resolved inside the safe directory.
- Agents can further scope access with `safe_dir` in frontmatter (relative to the safe root).
- If no safe directory is configured, all reads and writes are denied.

## Agent Search Order

The CLI merges user and system agents:

- User agents: `--agents-dir` (recursive).
- System agents: packaged in `quick_agent/agents` (recursive).

If agent ids collide, user agents win.

## Tool Search Order

The CLI merges user and system tools:

- User tools: `--tools-dir` (recursive).
- System tools: packaged in `quick_agent/tools` (recursive).

If tool ids collide, user tools win.
