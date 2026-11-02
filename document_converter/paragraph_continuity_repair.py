"""
Interrupted-paragraph repair pass (spec section 6, step 9: "Detect sentences
interrupted by unrelated regions").

`pdf_extractor.py`'s front-matter pass excises a front-matter-shaped line
*before* it ever joins a paragraph -- but when OCR merges a boilerplate
region (a permission notice, a DOI stamp) and an adjacent, unrelated body
region into one physical line with no detectable line break between them,
that line is already inside `paragraph_buf` by the time any per-line
classifier gets to see it, and the line-level check in pdf_extractor.py
never independently matches on the joined result. This is the residual
repair pass for exactly that case, run after extraction/normalization and
before rendering: it re-splits every paragraph into sentences and excises
any sentence that classifies as a strong front-matter kind, rejoining the
remaining sentences into one continuous paragraph so the reading order the
rest of the pipeline sees is the corrected one, not the original glued-together
text validation.py would otherwise have to block on.

Mirrors pdf_extractor.py's existing content policy for consistency: a
copyright/permission notice is dropped from the rendered body (real content
elsewhere already treats this as legitimately excludable boilerplate), while
a contact/conference/publication-metadata sentence is kept, just moved out
into its own block immediately after the paragraph it was excised from,
rather than staying jammed inside unrelated prose.

Kept fragments become `UnknownBlock`, not `ContactBlock`/`ConferenceMetadataBlock`/
`PublicationMetadataBlock` -- those front-matter-typed classes are, throughout
the rest of the pipeline (`validation.py`'s `_validate_front_matter_placement`,
`FRONT_MATTER_BLOCK_TYPES`), assumed to only ever appear in the one
contiguous group immediately after the title. A fragment recovered from the
*middle* of the body is not part of that group -- it just happens to be
boilerplate-shaped text that leaked into unrelated prose -- so giving it a
front-matter type would incorrectly trip "front matter is not grouped
contiguously after the title" for content that was never front matter to
begin with. `UnknownBlock` is exactly spec section 9's answer for this:
stays searchable, keeps source order and page provenance, is never promoted
to a heading, is never silently dropped.

Never applied inside a References/Bibliography section (see
`front_matter.is_references_heading`) -- a reference-list entry is expected
to look conference/DOI/author-shaped; that is correct, first-class content,
not an interruption to repair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .model import Document, ParagraphBlock, UnknownBlock
from .findings import FindingLog
from .front_matter import classify_front_matter_line, is_references_heading
from .chunker import build_sections

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_EXCLUDE_FROM_BODY = {"copyright", "permission"}
_STRONG_KINDS = frozenset({"contact", "conference", "publication_metadata", "copyright", "permission"})


@dataclass
class RepairStats:
    paragraphs_repaired: int = 0
    front_matter_sentences_excised: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "paragraphs_repaired": self.paragraphs_repaired,
            "front_matter_sentences_excised": self.front_matter_sentences_excised,
        }


def _references_section_block_ids(document: Document) -> set:
    ids: set = set()
    for s in build_sections(document):
        if any(is_references_heading(h) for h in s.header_path):
            ids.update(cb.block_id for cb in s.content_blocks)
    return ids


def repair_interrupted_paragraphs(document: Document, log: FindingLog) -> Dict[str, int]:
    """Mutates `document.blocks` in place. Returns repair-pass metrics."""
    stats = RepairStats()
    reference_ids = _references_section_block_ids(document)
    new_blocks: List = []
    # Seeded from the highest numeric block_id suffix actually in use (not
    # `len(document.blocks)` -- an earlier pass may have *removed* blocks,
    # e.g. `running_header_repair.py`, leaving remaining ids higher than the
    # current length) so a synthesized id here can never collide with one
    # still in use later in the list. Computed once, then incremented
    # locally rather than recalled per fragment -- `document.blocks` isn't
    # reassigned until this whole pass finishes, so `next_block_id()` itself
    # would keep returning the same value for every excised fragment.
    synthetic_counter = max(
        (int(m.group(1)) for m in (re.match(r"^b(\d+)$", b.block_id or "") for b in document.blocks) if m),
        default=0,
    )

    def next_synthetic_id() -> str:
        nonlocal synthetic_counter
        synthetic_counter += 1
        return f"b{synthetic_counter:04d}"

    for block in document.blocks:
        if not isinstance(block, ParagraphBlock) or block.block_id in reference_ids:
            new_blocks.append(block)
            continue

        text = block.text or block.normalized or ""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) < 2:
            new_blocks.append(block)
            continue

        kept: List[str] = []
        excised: List[tuple] = []  # (kind, sentence)
        for s in sentences:
            kind = classify_front_matter_line(s)
            if kind in _STRONG_KINDS:
                excised.append((kind, s))
            else:
                kept.append(s)

        if not excised or not kept:
            # Nothing to excise, or excising would leave no body text at all
            # (the whole paragraph is front matter shaped) -- leave it to
            # validation.py's checks rather than fabricating/deleting the
            # entire block here.
            new_blocks.append(block)
            continue

        block.text = " ".join(kept)
        block.normalized = block.text
        new_blocks.append(block)
        stats.paragraphs_repaired += 1

        for kind, sentence in excised:
            stats.front_matter_sentences_excised += 1
            if kind in _EXCLUDE_FROM_BODY:
                log.fixed(
                    "interrupting_front_matter_excised",
                    f"Excised embedded {kind} text from paragraph flow and dropped it per content "
                    f"policy (retained in audit log): {sentence!r}",
                    block_id=block.block_id, page=block.page,
                )
                continue
            typed_block = UnknownBlock(
                block_id=next_synthetic_id(), order=block.order, source_format=block.source_format,
                text=sentence, normalized=sentence, page=block.page,
            )
            new_blocks.append(typed_block)
            log.fixed(
                "interrupting_front_matter_excised",
                f"Excised embedded {kind} text from paragraph flow into its own block, "
                f"restoring paragraph continuity: {sentence!r}",
                block_id=typed_block.block_id, page=block.page,
            )

    document.blocks = new_blocks
    for i, b in enumerate(document.blocks):
        b.order = i
    return stats.to_dict()
