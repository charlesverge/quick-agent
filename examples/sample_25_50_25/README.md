# Sample 25-50-25 Example

This example demonstrates large-input sampling before the model call.

## Files

- `sample_agent.md`: agent config using sample ratios `25/50/25`.
- `input_6000_tokens.txt`: input fixture with more than 6000 whitespace tokens.

## Configuration Outline

The agent uses this content-processing configuration:

```yaml
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 2000
```

Behavior:

- Input larger than 2000 tokens is sampled into head/middle/footer windows.
- Ratio split is `25% / 50% / 25%`.
- The sampled text is capped to at most 2000 tokens.
- Model is `qwen3:0.6b` via local Ollama OpenAI-compatible endpoint.

## Run

From project root:

```bash
quick-agent \
  --agents-dir examples/sample_25_50_25 \
  --safe-dir . \
  --agent sample_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

Alternative module form:

```bash
python -m quick_agent.cli \
  --agents-dir examples/sample_25_50_25 \
  --safe-dir . \
  --agent sample_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

Output path:

- `out/sample_25_50_25_summary.json`
