from quick_agent.mapping.map_chunks import MapChunks
from quick_agent.mapping.map_paragraphs import MapParagraphs
from quick_agent.mapping.map_semchunk import MapSemchunk
from quick_agent.models.content_processing_spec import ChunkProcessingSpec


def test_map_chunks_splits_by_max_completion_tokens() -> None:
    mapper = MapChunks()
    config = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=3,
        overlap_percent=0,
    )

    chunks = mapper.run("one two three four five six seven", config)

    assert len(chunks) >= 2
    assert all(isinstance(entry, str) for entry in chunks)


def test_map_paragraphs_preserves_paragraphs_when_possible() -> None:
    mapper = MapParagraphs()
    config = ChunkProcessingSpec(
        mode="map_paragraphs",
        provider="semchunks",
        max_chunk_tokens=15,
        overlap_percent=0,
    )
    text = "p1 one two.\n\np2 three four.\n\np3 five six."

    chunks = mapper.run(text, config)

    assert len(chunks) >= 1
    assert "p1" in chunks[0]


def test_map_paragraphs_falls_back_for_long_paragraph() -> None:
    mapper = MapParagraphs()
    config = ChunkProcessingSpec(
        mode="map_paragraphs",
        provider="semchunks",
        max_chunk_tokens=4,
        overlap_percent=0,
    )
    text = "One two three four five six seven eight. Nine ten eleven twelve."

    chunks = mapper.run(text, config)

    assert len(chunks) >= 2


def test_overlap_token_precedence_over_percent() -> None:
    mapper = MapChunks()
    config_with_token = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=4,
        overlap_percent=20,
        overlap_token=2,
    )
    config_without_token = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=4,
        overlap_percent=20,
        overlap_token=None,
    )
    text = "one two three four five six seven eight nine ten"

    chunks_with_token = mapper.run(text, config_with_token)
    chunks_without_token = mapper.run(text, config_without_token)

    assert len(chunks_with_token) == len(chunks_without_token)
    if len(chunks_with_token) > 1:
        assert chunks_with_token[1] != chunks_without_token[1]


def test_map_semchunk_uses_mode() -> None:
    mapper = MapSemchunk()
    config = ChunkProcessingSpec(
        mode="map_paragraphs",
        provider="semchunks",
        max_chunk_tokens=8,
        overlap_percent=0,
    )
    text = "p1 one two three.\n\np2 four five six."

    chunks = mapper.run(text, config)

    assert len(chunks) >= 1


def test_max_output_items_limit() -> None:
    mapper = MapChunks()
    config = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=2,
        overlap_percent=0,
        max_output_items=2,
        max_output_tokens=None,
    )
    text = "one two three four five six seven eight"

    chunks = mapper.run(text, config)

    assert len(chunks) == 2


def test_max_output_tokens_limit() -> None:
    mapper = MapChunks()
    config = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=4,
        overlap_percent=0,
        max_output_items=30,
        max_output_tokens=5,
    )
    text = "one two three four five six seven eight"

    chunks = mapper.run(text, config)
    total_tokens = 0
    index = 0
    while index < len(chunks):
        total_tokens += len(mapper._encoding.encode(chunks[index]))
        index += 1

    assert total_tokens <= 5


def test_no_output_limits_when_both_none() -> None:
    mapper = MapChunks()
    config = ChunkProcessingSpec(
        mode="map_chunks",
        provider="semchunks",
        max_chunk_tokens=2,
        overlap_percent=0,
        max_output_items=None,
        max_output_tokens=None,
    )
    text = "one two three four five six seven eight"

    chunks = mapper.run(text, config)

    assert len(chunks) == 4
