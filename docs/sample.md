# Sample Content Processing

This document explains how to use `content_processing.sample` to reduce large inputs before model execution.

## What It Does

When `sample` is configured, the agent:

- Measures input using tokenizer tokens.
- Selects head/middle/footer token windows using configured ratios.
- Enforces a hard maximum token budget.
- Replaces runtime input text with the sampled text before normal chain execution.

## Configuration

```yaml
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 2000
    debug_output_file: out/sample_debug.txt
```

Fields:

- `ratios`: exactly three non-negative values for `head/middle/footer`.
- `max_chunk_tokens`: required maximum token budget; must be greater than zero.
- `debug_output_file`: optional path for writing sampled text for inspection.

Validation rules:

- Ratios must contain exactly three values.
- Every ratio value must be non-negative.
- Ratio sum must be greater than zero.
- `max_chunk_tokens` must be greater than zero.

## Example In Repo

- [examples/sample_25_50_25/README.md](../examples/sample_25_50_25/README.md)
- [examples/sample_25_50_25/sample_agent.md](../examples/sample_25_50_25/sample_agent.md)
- [examples/sample_25_50_25/input_6000_tokens.txt](../examples/sample_25_50_25/input_6000_tokens.txt)

## Run The Example

```bash
quick-agent \
  --agents-dir examples/sample_25_50_25 \
  --safe-dir . \
  --agent sample_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

Output file from the example agent:

- `out/sample_25_50_25_summary.json`
