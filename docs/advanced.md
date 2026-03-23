````markdown
# Advanced: extra_headers and extra_body

This document explains how to configure **extra HTTP headers** and **extra body parameters** for model requests when using the Python API (direct `QuickAgent` usage).

> These capabilities are not exposed via the CLI. They are intended for advanced embedding in other Python code.

## extra_headers

`extra_headers` is a dict of HTTP headers that will be included on every request made by the agent’s HTTP client.

This is useful for:

- Adding custom tracing headers
- Setting `Connection: close` in environments that require it
- Adding headers required by custom OpenAI-compatible providers

Example:

```python
from quick_agent.quick_agent import QuickAgent
from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput

registry = AgentRegistry(["agents"])
tools = AgentTools(["tools"])
permissions = DirectoryPermissions(Path("safe"))

agent = QuickAgent(
    registry=registry,
    tools=tools,
    directory_permissions=permissions,
    agent_id="my-agent",
    input_data=TextInput("Hello"),
    extra_headers={"X-My-Header": "my-value"},
)

result = await agent.run()
```

### Notes

- `extra_headers` is applied to the underlying `httpx.AsyncClient` used for model calls.
- Any headers you add will appear in every request (including to OpenAI and OpenAI-compatible endpoints).

## extra_body

`extra_body` is a dict that is merged into the model request payload via the pydantic-ai `ModelSettings.extra_body` mechanism.

This is useful for passing vendor-specific request options that are not part of the standard OpenAI request schema.

### num_ctx and OpenAI compatibility

When the agent is configured to use **OpenAI’s official endpoint** (`base_url: https://api.openai.com/v1`), the code explicitly **removes** `num_ctx` from `extra_body.options`.

That happens because OpenAI’s API does not accept `num_ctx`, but some OpenAI-compatible providers (e.g., Ollama) do.

So if you provide:

```python
extra_body={
    "options": {"num_ctx": 1024, "some_other_option": "value"}
}
```

- With **OpenAI** (`base_url=https://api.openai.com/v1`): `num_ctx` will be removed before the request is sent.
- With **OpenAI-compatible** endpoints (any other `base_url`): `num_ctx` will be passed through.

### Example (OpenAI-compatible provider)

When using an OpenAI-compatible endpoint (not `https://api.openai.com/v1`), QuickAgent also forces `format: json` in the request body to help ensure a JSON response.

```python
agent = QuickAgent(
    registry=registry,
    tools=tools,
    directory_permissions=permissions,
    agent_id="my-agent",
    input_data=TextInput("Hello"),
    extra_body={
        "options": {"num_ctx": 1024},
        "some_vendor_option": True,
    },
)
```

### Notes

- `extra_body` is merged into the running model settings and is used for both:
  - `pydantic-ai` model calls (tool-enabled runs)
  - OpenAI SDK structured single-shot runs (passed as `extra_body` to `openai` client)
- When using plain OpenAI, `num_ctx` is removed from `extra_body.options` to avoid rejection by OpenAI.

## ModelSpec defaults

`extra_headers` and `extra_body` can also be defined on `ModelSpec` (the configuration object used by the agent). The agent merges the model spec values with the constructor arguments; if the same keys exist in both places, the constructor values win.

```python
spec = ModelSpec(
    base_url="https://example.test/v1",
    model_name="gpt-test",
    extra_headers={"X-From-Model": "yes", "X-Shared": "model"},
    extra_body={"options": {"num_ctx": 512}},
)

agent = QuickAgent(
    ...,  # other required args
    model=spec,
    extra_headers={"X-From-Param": "yes", "X-Shared": "param"},
    extra_body={"options": {"num_ctx": 1024}},
)
```

## Empty agent body for preprocessing-only runs

An agent is treated as an empty body when all of these are true:

- `chain` is empty
- `instructions` is empty
- `system_prompt` is empty

In this mode, QuickAgent does not call the LLM. The returned output is only the
result of content preprocessing.

### Sample only

```markdown
---
name: "Sample only"
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 2000
chain: []
---
```

Result: returns sampled text as the final output.

### Chunk only

```markdown
---
name: "Chunk only"
content_processing:
  chunk_processing:
    mode: map_chunks
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
chain: []
---
```

Result: returns `{"items": [...]}` where each item is chunk text.

### Sample then chunk

```markdown
---
name: "Sample then chunk"
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 4000
  chunk_processing:
    mode: map_paragraphs
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
chain: []
---
```

Result: sample is applied first, then chunking, and output is `{"items": [...]}`.

````
