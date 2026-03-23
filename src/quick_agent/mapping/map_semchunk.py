"""Semchunks provider mapper."""

from __future__ import annotations

from quick_agent.mapping.map_base import MapBase
from quick_agent.mapping.map_chunks import MapChunks
from quick_agent.mapping.map_paragraphs import MapParagraphs
from quick_agent.models.content_processing_spec import ChunkProcessingSpec


class MapSemchunk(MapBase):
    def build_chunks(self, text: str, map_config: ChunkProcessingSpec) -> list[str]:
        if map_config.mode == "map_chunks":
            return MapChunks().build_chunks(text, map_config)
        return MapParagraphs().build_chunks(text, map_config)
