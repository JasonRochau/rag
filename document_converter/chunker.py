"""
Canonical Markdown chunker (spec section 9).

One implementation, used for validation, chunk preview, real ingestion and
tests -- there is no separate "preview" algorithm. It walks the heading
tree built from a normalized `Document`, treats lists/tables/code as
atomic, packs paragraph-sized blocks toward `target_words` without ever
exceeding the effective hard limit (which already accounts for the
provenance prefix and heading overhead), and routes any paragraph that
still exceeds the budget through the emergency sentence-boundary split --
in EVERY profile, strict included; a single sentence that alone exceeds
the real token budget is further force-split at token boundaries via
`atomic_split._forced_split_line`, so one long sentence can no longer
trip the final `chunk_exceeds_token_limit` guard.

Atomic blocks (list/table/code/...) are never split arbitrarily, mid-line,
or without preserving structural provenance -- but one that exceeds the
effective chunk budget is no longer a hard stop. It is handed to
`atomic_split.split_oversized_atomic_block`, which subdivides it at safe
semantic boundaries (function/algorithm boundaries and blank lines before
falling back to per-line packing, forcing a mid-line split only when a
single line itself is oversized) and returns token-verified parts, each
emitted as its own chunk. This is the one splitting implementation shared
by preview/process (approximate tokenizer) and index (the live embedding
service's real tokenizer) -- see atomic_split.py's module docstring.

A block that genuinely cannot be subdivided within budget (`AtomicSplitError`
-- e.g. a table whose key cell alone fills the budget) is DROPPED from the
index with a WARNING (`oversized_atomic_block`), never an error: one bad
block must not fail the whole document's upload.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .model import Document, Block, HeadingBlock, ParagraphBlock
from .render import render_block, render_heading
from .findings import FindingLog
from .config import ChunkingConfig
from .atomic_split import split_oversized_atomic_block, AtomicSplitError, _forced_split_line

# Document is the canonical parsed-markdown representation (see md_parser.py /
# render.py docstrings for why no separate re-parse step is needed). Aliased
# here to match the interface name used in the spec.
ParsedMarkdownDocument = Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def _approx_token_count(text: str) -> int:
    """Offline fallback estimator (~1.3 tokens/word) used when no real tokenizer is wired in."""
    words = len(text.split())
    return int(round(words * 1.3))


@dataclass
class Section:
    header_path: List[str]
    heading_block: Optional[HeadingBlock]
    content_blocks: List[Block] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_index: int
    header_path: List[str]
    heading: str
    provenance: str
    text: str
    body: str
    word_count: int
    token_count: int
    source_block_ids: List[str]
    source_pages: List[int]
    # Distinct block-class names contributing to this chunk (e.g.
    # ["ParagraphBlock"], ["ListBlock", "ParagraphBlock"]) -- the "block
    # type or content type" minimal-metadata field a MAIN_TEXT chunk needs
    # (spec Phase 3). A chunk can span more than one block, so this is a
    # list, not a single type; order is source order, not sorted, so it
    # also hints at composition (e.g. a caption-then-paragraph pair) at a
    # glance in the index JSON.
    content_types: List[str] = field(default_factory=list)
    forced_fallback: bool = False
    # Populated only for a chunk that is one part of a subdivided oversized
    # atomic block (spec section 4's parent/child provenance schema); None
    # for every ordinary chunk.
    split_provenance: Optional[Dict[str, Any]] = None


@dataclass
class ChunkingResult:
    chunks: List[ChunkRecord] = field(default_factory=list)
    findings: FindingLog = field(default_factory=FindingLog)
    forced_fallback_splits: int = 0
    # Oversized atomic blocks that were *successfully* subdivided into
    # token-safe parts (not dropped -- every part was emitted as its own
    # chunk). Kept distinct from `oversized_atomic_blocks` below, which now
    # only counts the (expected to be vanishingly rare) case where
    # subdivision itself could not bring every part under budget.
    atomic_blocks_split: int = 0
    forced_split_atomic_parts: int = 0
    # Count of oversized atomic blocks that subdivision could not resolve
    # (AtomicSplitError). Such a block is DROPPED from the index with a
    # WARNING -- not an error -- so the document itself still succeeds.
    # Distinct from `atomic_blocks_split` above. Spec section 10 calls this
    # `oversized_unresolved_blocks`.
    oversized_atomic_blocks: int = 0
    dropped_block_ids: List[str] = field(default_factory=list)
    # parent_block_id -> ordered list of the part block_ids it was split
    # into, so validation can verify every part is represented exactly once
    # instead of expecting the (no-longer-existing) single parent id.
    split_block_provenance: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def largest_chunk_words(self) -> int:
        return max((c.word_count for c in self.chunks), default=0)

    @property
    def largest_chunk_tokens(self) -> int:
        return max((c.token_count for c in self.chunks), default=0)


def build_sections(document: Document) -> List[Section]:
    stack: List[HeadingBlock] = []
    sections: List[Section] = []
    current = Section(header_path=[], heading_block=None, content_blocks=[])

    for block in document.blocks:
        if isinstance(block, HeadingBlock):
            sections.append(current)
            while stack and stack[-1].level >= block.level:
                stack.pop()
            stack.append(block)
            current = Section(header_path=[h.text for h in stack], heading_block=block, content_blocks=[])
        else:
            current.content_blocks.append(block)
    sections.append(current)
    return [s for s in sections if s.heading_block is not None or s.content_blocks]


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _emergency_split_paragraph(
    text: str, max_words: int,
    token_counter: Optional[Callable[[str], int]] = None, max_tokens: Optional[int] = None,
) -> List[str]:
    """Splits a legitimately long paragraph on sentence boundaries -- never
    mid-sentence, never mid-word -- packing sentences toward `max_words`.
    When a real tokenizer is available (`token_counter`/`max_tokens`, wired
    in by the index path -- see chunk()'s token-aware packing), each
    candidate piece is also checked against the real token budget, not just
    the word-count proxy, since technical academic prose routinely tokenizes
    at well over 1 token/word. A single sentence that alone exceeds either
    budget still comes out of here whole (sentences are never split) -- the
    CALLER then runs such pieces through `atomic_split._forced_split_line`
    for a token-level forced split, so an oversized sentence can no longer
    reach the final `chunk_exceeds_token_limit` guard intact."""
    sentences = _split_sentences(text)
    pieces: List[str] = []
    current: List[str] = []
    current_words = 0
    for sent in sentences:
        w = len(sent.split())
        over_words = current and current_words + w > max_words
        over_tokens = (
            current and token_counter is not None and max_tokens is not None
            and token_counter(" ".join(current + [sent])) > max_tokens
        )
        if current and (over_words or over_tokens):
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sent)
        current_words += w
    if current:
        pieces.append(" ".join(current))
    return pieces


class MarkdownChunker:
    """The one canonical chunker implementation."""

    def chunk(
        self,
        document: ParsedMarkdownDocument,
        source_path: str,
        config: ChunkingConfig,
        document_id: str = "",
        token_counter: Optional[Callable[[str], int]] = None,
        max_sequence_tokens: Optional[int] = None,
    ) -> ChunkingResult:
        # `token_counter`/`max_sequence_tokens` are only used to drive packing
        # decisions when both are supplied by a real embedding service (index()
        # re-chunks with these wired in -- see pipeline.py). Word count is a
        # poor proxy for token count on technical content (CLI syntax, code,
        # hyphenated jargon routinely tokenizes at ~2x the word count), so
        # relying on a word budget alone under-packs or over-packs relative to
        # the model's real limit. preview()/process() have no embedding
        # service and fall back to word-only packing plus the offline
        # approximator for the reported token_count.
        pack_token_counter = token_counter if (token_counter is not None and max_sequence_tokens) else None
        report_token_counter = token_counter or _approx_token_count
        result = ChunkingResult()
        sections = build_sections(document)
        filename = source_path.rsplit("/", 1)[-1] if source_path else "document.md"
        chunk_index = 0

        config_fp = _config_fingerprint(config)

        for section in sections:
            heading_text = section.heading_block.text if section.heading_block else filename
            path_for_provenance = section.header_path or [heading_text]
            provenance = f"[{filename} > {' > '.join(path_for_provenance)}]"
            heading_line = render_heading(section.heading_block) if section.heading_block else f"# {heading_text}"

            overhead_words = len(provenance.split()) + len(heading_line.split())
            effective_hard = max(config.hard_max_words - overhead_words, 1)
            effective_target = max(config.target_words - overhead_words, 1)

            effective_hard_tokens = None
            effective_target_tokens = None
            if pack_token_counter is not None:
                overhead_tokens = pack_token_counter(provenance) + pack_token_counter(heading_line)
                # `config.max_tokens` (when set -- e.g. a Qwen-configured
                # pipeline's EMBEDDING_CHUNK_MAX_TOKENS) is an *operating*
                # ceiling, never allowed to exceed what the live embedding
                # service actually accepts (`max_sequence_tokens`) -- the
                # smaller of the two always wins, so a misconfigured
                # operating max can never produce a chunk the service would
                # reject. `None` (unset -- BGE/legacy) falls back to the
                # service's own reported limit exactly as before.
                hard_ceiling = min(config.max_tokens, max_sequence_tokens) if config.max_tokens else max_sequence_tokens
                effective_hard_tokens = max(hard_ceiling - overhead_tokens, 1)
                if config.target_tokens:
                    # Token-native target (Qwen path): used directly, still
                    # implicitly bounded by `effective_hard_tokens` above --
                    # packing flushes at the hard limit regardless, so this
                    # is a preferred size to aim for, not a size to pad to
                    # (spec: "a chunk may be smaller when that is the
                    # natural semantic boundary").
                    effective_target_tokens = max(config.target_tokens - overhead_tokens, 1)
                else:
                    # Legacy BGE/word-based path: target scales
                    # proportionally with the hard limit, exactly as before.
                    ratio = (config.target_words / config.hard_max_words) if config.hard_max_words else 0.85
                    effective_target_tokens = max(int(effective_hard_tokens * ratio), 1)

            packed_groups = self._pack_blocks(
                section, effective_hard, effective_target, config, result,
                token_counter=pack_token_counter,
                effective_hard_tokens=effective_hard_tokens,
                effective_target_tokens=effective_target_tokens,
                document_id=document_id,
            )

            for group in packed_groups:
                if not group:
                    continue
                block_fragments = []
                for b in group:
                    ctx_header = b.style.get("atomic_split_context_header")
                    rendered = render_block(b)
                    block_fragments.append(f"{ctx_header}\n\n{rendered}" if ctx_header else rendered)
                body_fragments = [heading_line] + block_fragments
                body = "\n\n".join(body_fragments)
                text = f"{provenance}\n\n{body}" if config.include_provenance_prefix else body
                word_count = len(text.split())
                token_count = report_token_counter(text)
                if max_sequence_tokens and token_count > max_sequence_tokens:
                    result.findings.error(
                        "chunk_exceeds_token_limit",
                        f"Chunk under {provenance} has {token_count} tokens, exceeding the embedding "
                        f"model's max sequence length of {max_sequence_tokens}",
                    )
                block_ids = [b.block_id for b in group]
                pages = sorted({b.page for b in group if b.page is not None})
                content_types = list(dict.fromkeys(type(b).__name__ for b in group))
                chunk_id = _stable_chunk_id(document_id, path_for_provenance, text, config_fp)
                forced = any(getattr(b, "_forced_fallback", False) for b in group)
                split_provenance = None
                if len(group) == 1:
                    split_prov = group[0].style.get("atomic_split_provenance")
                    if split_prov is not None:
                        split_provenance = dict(split_prov, document_id=document_id)
                result.chunks.append(ChunkRecord(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    header_path=list(path_for_provenance),
                    heading=heading_text,
                    provenance=provenance,
                    text=text,
                    body=body,
                    word_count=word_count,
                    token_count=token_count,
                    source_block_ids=block_ids,
                    source_pages=pages,
                    content_types=content_types,
                    forced_fallback=forced,
                    split_provenance=split_provenance,
                ))
                chunk_index += 1

        result.forced_fallback_splits = sum(1 for c in result.chunks if c.forced_fallback)
        return result

    def _pack_blocks(self, section: Section, effective_hard: int, effective_target: int,
                      config: ChunkingConfig, result: ChunkingResult,
                      token_counter: Optional[Callable[[str], int]] = None,
                      effective_hard_tokens: Optional[int] = None,
                      effective_target_tokens: Optional[int] = None,
                      document_id: str = "") -> List[List[Block]]:
        groups: List[List[Block]] = []
        current: List[Block] = []
        current_words = 0
        current_tokens = 0
        token_aware = token_counter is not None and effective_hard_tokens is not None

        def flush():
            nonlocal current, current_words, current_tokens
            if current:
                groups.append(current)
            current, current_words, current_tokens = [], 0, 0

        for block in section.content_blocks:
            bwc = block.word_count
            # Summing each block's independently-tokenized count is a
            # conservative (>=) estimate of the concatenated group's real
            # token count, and needs exactly one token_counter call per block
            # regardless of how packing groups it -- see chunk()'s docstring
            # note on why word count alone is not a reliable proxy here.
            btc = token_counter(render_block(block)) if token_aware else 0
            oversized_tokens = token_aware and btc > effective_hard_tokens

            if bwc > effective_hard or oversized_tokens:
                budget_desc = f"{effective_hard} words"
                if token_aware:
                    budget_desc += f" / {effective_hard_tokens} tokens"
                if isinstance(block, ParagraphBlock):
                    flush()
                    # Oversized paragraphs route through the emergency
                    # sentence-boundary split in EVERY profile, strict
                    # included -- the old strict-mode error+drop branch
                    # discarded real body text and failed whole documents on
                    # one long paragraph, which is no longer the policy.
                    pieces = _emergency_split_paragraph(
                        block.text, effective_hard,
                        token_counter=token_counter if token_aware else None,
                        max_tokens=effective_hard_tokens if token_aware else None,
                    )
                    # A single sentence that alone still exceeds the real
                    # token budget (sentence splitting can't help it) gets a
                    # token-level forced split via the same mid-line splitter
                    # atomic code blocks use -- otherwise it would trip the
                    # final `chunk_exceeds_token_limit` guard and fail the
                    # whole document in strict mode.
                    if token_aware:
                        safe_pieces: List[str] = []
                        for piece in pieces:
                            if token_counter(piece) > effective_hard_tokens:
                                safe_pieces.extend(
                                    _forced_split_line(piece, effective_hard_tokens, token_counter)
                                )
                            else:
                                safe_pieces.append(piece)
                        pieces = safe_pieces
                    result.findings.warning(
                        "forced_fallback_split",
                        f"Paragraph ({bwc} words) exceeded the budget; force-split into {len(pieces)} "
                        f"piece(s) (sentence boundaries first, then a token-level forced split for "
                        f"any single sentence still over the token budget)",
                        block_id=block.block_id,
                    )
                    # Each piece becomes its own chunk (see the `groups.append`
                    # below), so it needs its own id distinct from the parent's
                    # -- reusing `block.block_id` on every piece would make
                    # `_validate_source_coverage` see the same source block id
                    # "duplicated" across N chunks instead of correctly split
                    # into N parts (mirrors `split_oversized_atomic_block`'s
                    # parent/part provenance convention exactly).
                    part_ids = [f"{block.block_id}-part-{i + 1}" for i in range(len(pieces))]
                    result.split_block_provenance[block.block_id] = part_ids
                    for part_id, piece in zip(part_ids, pieces):
                        mini = ParagraphBlock(
                            block_id=part_id, order=block.order, source_format=block.source_format,
                            page=block.page, text=piece, normalized=piece,
                        )
                        mini._forced_fallback = True
                        groups.append([mini])
                else:
                    kind = type(block).__name__.replace("Block", "").lower()
                    split_tokenizer = token_counter if token_aware else _approx_token_count
                    split_budget = effective_hard_tokens if token_aware else int(round(effective_hard * 1.3))
                    try:
                        parts = split_oversized_atomic_block(
                            block, tokenizer=split_tokenizer, max_tokens=split_budget,
                            document_id=document_id, page_start=block.page, page_end=block.page,
                            oversized_row_policy=config.oversized_row_policy,
                        )
                    except AtomicSplitError as e:
                        # Agreed policy: a block that truly cannot be split
                        # within budget is DROPPED WITH A WARNING; the
                        # document succeeds. One bad table must never fail
                        # an entire upload (the old error-level finding did
                        # exactly that under strict mode).
                        result.oversized_atomic_blocks += 1
                        result.findings.warning(
                            "oversized_atomic_block",
                            f"{kind} block ({bwc} words" + (f", {btc} tokens" if token_aware else "") +
                            f") exceeds the effective chunk budget ({budget_desc}) and could not be "
                            f"safely subdivided: {e}",
                            block_id=block.block_id,
                            suggested_action=f"Block was dropped from the index; shorten the {kind} "
                            f"at the source or raise hard_max_words to retain it.",
                        )
                        result.dropped_block_ids.append(block.block_id)
                        continue

                    flush()
                    result.atomic_blocks_split += 1
                    result.split_block_provenance[block.block_id] = [p.block.block_id for p in parts]
                    forced_here = sum(1 for p in parts if p.forced_split)
                    result.forced_split_atomic_parts += forced_here
                    result.findings.warning(
                        "oversized_atomic_block_subdivided",
                        f"{kind} block ({bwc} words" + (f", {btc} tokens" if token_aware else "") +
                        f") exceeds the effective chunk budget ({budget_desc}); subdivided into "
                        f"{len(parts)} token-safe part(s)" + (f", {forced_here} forced" if forced_here else ""),
                        block_id=block.block_id,
                    )
                    for p in parts:
                        p.block.style["atomic_split_provenance"] = dict(p.provenance)
                        if p.forced_split:
                            result.findings.warning(
                                "forced_long_line_subdivision",
                                f"Part {p.part_index}/{p.part_count} of {kind} block {block.block_id} "
                                f"required a forced mid-line split (a single source line alone exceeded "
                                f"the effective budget); full original line preserved in reconstruction "
                                f"metadata, not truncated",
                                block_id=p.block.block_id,
                            )
                        groups.append([p.block])
                continue

            would_exceed = current_words + bwc > effective_hard or (
                token_aware and current_tokens + btc > effective_hard_tokens
            )
            if would_exceed and current:
                flush()
            current.append(block)
            current_words += bwc
            current_tokens += btc
            target_reached = current_words >= effective_target or (
                token_aware and effective_target_tokens is not None and current_tokens >= effective_target_tokens
            )
            if target_reached:
                flush()

        flush()
        return groups


def _config_fingerprint(config: ChunkingConfig) -> str:
    payload = repr(sorted(config.fingerprint_fields().items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _stable_chunk_id(document_id: str, header_path: List[str], text: str, config_fp: str) -> str:
    payload = "\x1f".join([document_id, "\x1e".join(header_path), text, config_fp])
    return "c-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
