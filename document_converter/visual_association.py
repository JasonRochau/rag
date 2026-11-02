"""
Associates a `VisualRegion` with its surrounding MAIN_TEXT context (spec
"Phase 3") -- targeted context expansion, never document reconstruction.

Reuses the MAIN_TEXT chunks already produced by `MarkdownChunker` for this
same document (`ChunkRecord.source_pages`, `.header_path`, `.chunk_index`,
`.chunk_id`) instead of re-deriving position/section information from the
PDF a third time. A visual's page number (from `visual_extraction.py`) is
matched against each chunk's `source_pages` to find same-page prose, and
`chunk_index` order is used to find the nearest preceding/following chunk
when no chunk shares the exact page (common for a full-page figure with the
describing prose on the page before or after it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .chunker import ChunkRecord


@dataclass
class VisualAssociation:
    heading_path: List[str]
    linked_chunk_ids: List[str]


# Cap on how many MAIN_TEXT chunks one visual links to -- context expansion,
# not document reconstruction (spec: "do not solve association by dumping
# entire pages/documents into each visual record").
MAX_LINKED_CHUNKS = 4


def associate_visual(page: int, chunks: Sequence[ChunkRecord]) -> VisualAssociation:
    """`chunks` must be in document order (as `MarkdownChunker.chunk()`
    already produces them)."""
    if not chunks:
        return VisualAssociation(heading_path=[], linked_chunk_ids=[])

    same_page = [c for c in chunks if page in (c.source_pages or [])]
    linked: List[ChunkRecord] = list(same_page[:MAX_LINKED_CHUNKS])

    if not linked:
        # No chunk shares this exact page (e.g. a page that is entirely one
        # full-page figure) -- fall back to the nearest preceding and
        # following chunk by page number, which is the closest available
        # proxy for "what section is this visual physically inside".
        preceding = [c for c in chunks if all(p <= page for p in (c.source_pages or [page]))]
        following = [c for c in chunks if any(p >= page for p in (c.source_pages or [page]))]
        if preceding:
            linked.append(preceding[-1])
        if following and (not preceding or following[0].chunk_id != preceding[-1].chunk_id):
            linked.append(following[0])

    linked = linked[:MAX_LINKED_CHUNKS]
    heading_path = list(linked[0].header_path) if linked else []
    # De-duplicate while preserving order (same_page + preceding/following
    # can never actually collide given the branches above, but a defensive
    # dedupe keeps this correct if that changes).
    seen = set()
    linked_ids: List[str] = []
    for c in linked:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            linked_ids.append(c.chunk_id)

    return VisualAssociation(heading_path=heading_path, linked_chunk_ids=linked_ids)
