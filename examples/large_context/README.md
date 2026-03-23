# Large Context Chunking Examples

This example set demonstrates large-input chunk processing with map modes and
sample-then-chunk flow.

## Files

- `map_chunks_agent.md`: token-size chunking with `mode: map_chunks`.
- `map_paragraphs_agent.md`: paragraph-first chunking with `mode: map_paragraphs`.
- `sample_then_chunk_agent.md`: applies sample first, then paragraph chunking.
- `../sample_25_50_25/input_6000_tokens.txt`: input fixture with more than 6000 whitespace tokens.

## Configuration Outline

All examples use:

- `provider: semchunks`
- output format `json`
- local OpenAI-compatible model endpoint (`http://localhost:11434/v1`)

### map_chunks config

```yaml
content_processing:
  chunk_processing:
    mode: map_chunks
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
```

### map_paragraphs config

```yaml
content_processing:
  chunk_processing:
    mode: map_paragraphs
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
```

### sample then chunk config

```yaml
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 4000
  chunk_processing:
    mode: map_paragraphs
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
```

## Run

From project root:

### run map_chunks

```bash
quick-agent \
  --agents-dir examples/large_context \
  --safe-dir . \
  --agent map_chunks_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

### run map_paragraphs

```bash
quick-agent \
  --agents-dir examples/large_context \
  --safe-dir . \
  --agent map_paragraphs_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

### run sample then chunk

```bash
quick-agent \
  --agents-dir examples/large_context \
  --safe-dir . \
  --agent sample_then_chunk_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

Alternative module form:

```bash
python -m quick_agent.cli \
  --agents-dir examples/large_context \
  --safe-dir . \
  --agent map_chunks_agent \
  --input examples/sample_25_50_25/input_6000_tokens.txt
```

## Output paths

- `out/large_context_map_chunks.json`
- `out/large_context_map_paragraphs.json`
- `out/large_context_sample_then_chunk.json`
