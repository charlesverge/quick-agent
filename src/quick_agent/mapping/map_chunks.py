"""Token-size chunk mapper."""

from __future__ import annotations

from quick_agent.mapping.map_base import MapBase
from quick_agent.models.content_processing_spec import ChunkProcessingSpec


class MapChunks(MapBase):
    def build_chunks(self, text: str, map_config: ChunkProcessingSpec) -> list[str]:
        token_ids = self._encoding.encode(text)
        return self._split_token_ids(token_ids, map_config.max_chunk_tokens)
