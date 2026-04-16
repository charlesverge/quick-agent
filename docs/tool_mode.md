# Tool Mode

## Overview

`tool_mode` controls how pydantic\_ai interacts with the LLM for structured
output and tool calling.  The setting exists because Ollama's
OpenAI-compatible API rejects `"content": null` in assistant messages — a
value that pydantic\_ai sets when the model responds with only tool calls and
no text content.

## The root cause

When pydantic\_ai uses tool mode for structured output (the default), it
registers a synthetic `final_result` tool.  The model calls that tool and
the assistant message in the conversation history carries the tool call but
no text.  pydantic\_ai serialises this as:

```json
{ "role": "assistant", "content": null, "tool_calls": [...] }
```

Ollama's Go server treats `null` as Go's `nil` and returns:

```text
400 Bad Request — invalid message content type: <nil>
```

This happens in **two** situations:

1. **No user-defined tools, BaseModel output.**
   pydantic\_ai's tool-mode output creates a multi-turn conversation
   (request → tool call response → follow-up) where the assistant message
   has `content: null`.

2. **User-defined tools with multi-turn conversations.**
   The model calls a real tool first.  The assistant message for that turn
   has `content: null` because the model produced only a tool call.
   The next request includes that message and Ollama rejects it.

## The four modes

| Mode | Model class | Profile | When to use |
|---|---|---|---|
| `default` | `OpenAIChatModel` | none | OpenAI API, or providers that accept `content: null` |
| `no_tools` | `OllamaSafeChatModel` | prompted + json\_object | Ollama agents **without** tools — avoids tool calling entirely |
| `with_tools` | `OllamaSafeChatModel` | strict=off | Ollama agents **with** tools — patches `content: null` → `""` |
| `prompted_tools` | `OpenAIChatModel` | prompted + json\_object | Experimental — prompted output + tools (may confuse some models) |

### `default`

Standard pydantic\_ai behaviour.  Uses tool mode for structured output.
No custom profile.  Works with OpenAI and providers that accept
`content: null` in assistant messages.

### `no_tools`

Best for Ollama agents that have no tools defined.

- Uses `OllamaSafeChatModel` (patches `content: null` → `""` as a safety
  net).
- Sets `default_structured_output_mode='prompted'` and
  `supports_json_object_output=True` on the profile so pydantic\_ai asks
  for JSON in the prompt instead of using a `final_result` tool.
- Avoids the tool-calling mechanism entirely — the model produces JSON
  directly in its text response.

### `with_tools`

Required for Ollama agents that define tools.

- Uses `OllamaSafeChatModel` which overrides `_into_message_param()` to
  replace `content: null` with `content: ""` in assistant messages.
- Keeps the default tool-mode structured output so `final_result` works.
- Sets `openai_supports_strict_tool_definition=False` because Ollama does
  not support strict tool schemas.

### `prompted_tools`

Experimental.  Combines prompted structured output with tool calling.

- Uses standard `OpenAIChatModel` (no subclass).
- Sets `default_structured_output_mode='prompted'` so structured output
  uses JSON-in-text instead of `final_result` tool.
- Tools are still registered normally.
- **Warning:** some models get confused when they receive both a tool list
  and a prompt asking for JSON output.  Test thoroughly before using this
  mode in production.

## Test results (qwen2.5-7b-16k via Ollama)

| Scenario | default | no\_tools | with\_tools | prompted\_tools |
|---|---|---|---|---|
| BaseModel, no tools | FAIL | **PASS** | FAIL (validation) | **PASS** |
| BaseModel, with tools | PASS | n/a | **PASS** | FAIL |
| Forced multi-turn tool call | FAIL | n/a | **PASS** | FAIL |

## Agent template examples

### No tools — extraction agent (recommended: `no_tools`)

```yaml
---
name: "company_extract"
description: "Extract company data from text."
tool_mode: "no_tools"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "qwen2.5-7b-16k:latest"
schemas:
  CompanyData: "myapp.schemas:CompanyData"
chain:
  - id: extract
    kind: structured
    prompt_section: step:extract
    output_schema: CompanyData
output:
  format: "json"
  file: "out/company.json"
---

## Instructions

Extract the company name, location, and industry from the input text.

## step:extract

Parse the input and return structured company data.
```

### With tools — file manager agent (recommended: `with_tools`)

```yaml
---
name: "file_manager"
description: "Manage files using filesystem tools."
tool_mode: "with_tools"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "qwen2.5-7b-16k:latest"
tools:
  - "filesystem.list_files"
  - "filesystem.read_text"
  - "filesystem.write_text"
chain:
  - id: execute
    kind: text
    prompt_section: step:execute
output:
  format: "markdown"
  file: "out/result.md"
---

## Instructions

You are a file management assistant.

## step:execute

Complete the requested file operation.
```

### OpenAI API — no tool\_mode needed (default)

```yaml
---
name: "summarizer"
description: "Summarize text using OpenAI."
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-4o"
schemas:
  Summary: "myapp.schemas:Summary"
chain:
  - id: summarize
    kind: structured
    prompt_section: step:summarize
    output_schema: Summary
output:
  format: "json"
  file: "out/summary.json"
---

## Instructions

Summarize the input text.

## step:summarize

Return a structured summary.
```
