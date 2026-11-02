"""
MAIN_TEXT retrieval view (see DELIVERABLES.md, "MAIN_TEXT vs FULL_FIDELITY
ingestion profiles").

Derives a filtered `Document` from an already fully-extracted and
normalized one -- no re-parsing, no second extraction pipeline, no new
block types. Every block class and repair pass this module reads already
exists (`document_converter/model.py`, `pdf_extractor.py`, `normalize.py`,
...); this module only decides which of the already-classified blocks
belong in the index used for semantic search, and which of them a title
may legitimately be chosen from.

FULL_FIDELITY mode never calls into this module at all -- it chunks and
validates the original, unfiltered `Document` exactly as before this
profile system existed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .model import (
    Block, Document, HeadingBlock, ParagraphBlock, ListBlock, CodeBlock, TableBlock,
    QuoteBlock, UnknownBlock, WarningBlock, CaptionBlock, FigureLabelBlock,
    DiagramAnnotationBlock, TableCellBlock, ImageBlock, FRONT_MATTER_BLOCK_TYPES,
)
from .front_matter import (
    is_references_heading, is_toc_heading, is_abstract_heading, is_acknowledgements_heading,
    classify_front_matter_line,
)
from .extractors import heading_rejection_reason, numbering_tuple

# Block types that are never main-text content, regardless of where they
# appear (spec "EXCLUDE" list): front matter (author/affiliation/contact/
# keywords/publication-copyright-permission-conference metadata) plus every
# graphical/figure-adjacent typed block already produced by the extractor.
# A real `TableBlock` is deliberately *not* in this set -- existing table
# support is left exactly as it is, per this task's "do not delete table
# support" constraint. `TableCellBlock` (an ungrouped, loose cell fragment
# rather than a real assembled table) is graphical/layout debris for
# retrieval purposes and is excluded.
EXCLUDED_BLOCK_TYPES = FRONT_MATTER_BLOCK_TYPES + (
    CaptionBlock, FigureLabelBlock, DiagramAnnotationBlock, TableCellBlock, ImageBlock,
)

CONTENT_BLOCK_TYPES = (ParagraphBlock, ListBlock, CodeBlock, TableBlock, QuoteBlock, UnknownBlock, WarningBlock)


@dataclass
class MainTextFilterStats:
    """Per-category counts of what `build_main_text_document` removed, for
    the diagnostic/report side (spec Phase 9's "excluded blocks by
    category")."""
    excluded_by_type: Dict[str, int] = field(default_factory=dict)
    excluded_references_blocks: int = 0
    excluded_toc_blocks: int = 0
    excluded_shell_heading_blocks: int = 0
    excluded_abstract_blocks: int = 0
    excluded_acknowledgements_blocks: int = 0
    excluded_junk_heading_blocks: int = 0
    excluded_front_matter_zone_blocks: int = 0
    total_source_blocks: int = 0
    included_blocks: int = 0
    # The title text (if any) reconstructed from short, un-typed paragraph
    # fragments in the document's front-matter zone -- e.g. a wrapped
    # multi-line title that never got reassembled into one TitleBlock/
    # HeadingBlock at extraction time. Not a count; carried through here so
    # `resolve_main_text_title` (which needs it) and diagnostics (which want
    # to see it) share the one place it's computed.
    title_region_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "excluded_by_type": dict(self.excluded_by_type),
            "excluded_references_blocks": self.excluded_references_blocks,
            "excluded_toc_blocks": self.excluded_toc_blocks,
            "excluded_shell_heading_blocks": self.excluded_shell_heading_blocks,
            "excluded_abstract_blocks": self.excluded_abstract_blocks,
            "excluded_acknowledgements_blocks": self.excluded_acknowledgements_blocks,
            "excluded_junk_heading_blocks": self.excluded_junk_heading_blocks,
            "excluded_front_matter_zone_blocks": self.excluded_front_matter_zone_blocks,
            "total_source_blocks": self.total_source_blocks,
            "included_blocks": self.included_blocks,
            "title_region_text": self.title_region_text,
        }


def _is_shell_heading(blocks: List[Block], idx: int, first_heading_idx: Optional[int]) -> bool:
    """A heading immediately followed by a block of an excluded (figure/
    caption/publication-metadata/...) type is figure or citation debris
    that happened to get promoted to a heading, not a real section title --
    the neighboring-block evidence Phase 5 asks for, rather than a string
    match against the heading's own text (a figure label such as "Modern
    processor" must never become a title or section heading, but the fix
    has to generalize past that one string).

    Exempts the document's *first* heading unconditionally: a title
    immediately followed by front matter (author/affiliation/contact/
    publication metadata) is completely ordinary academic-paper structure,
    not evidence the title itself is debris -- only a *later*, non-title
    heading immediately preceding excluded content is the suspicious
    pattern this rule exists to catch."""
    if idx == first_heading_idx:
        return False
    j = idx + 1
    if j >= len(blocks):
        return False
    return isinstance(blocks[j], EXCLUDED_BLOCK_TYPES)


# heading_rejection_reason (extractors/__init__.py) is the same text-shape
# classifier already trusted, "regardless of source format or extraction
# path" (its own docstring), to keep code/equation/caption/page-furniture
# lines out of HeadingBlock in the first place, and re-run retrospectively
# by validation.py's `_validate_heading_quality` against every heading that
# *did* make it through. Reusing it here is the same "re-run an existing
# classifier against already-classified content" pattern as
# `_is_shell_heading` below -- not a new heuristic, and not something that
# needs re-deriving per document format, since it already had to work
# across all of them. Restricted to the reasons that specifically mean
# "this is visual-region/non-prose debris, not body text with unusual
# phrasing" (spec EXCLUDE: "standalone equations and symbol debris",
# "diagrams and diagram labels", "table headings", "captions", plus the
# spec's own "do not promote algorithms ... into document headings") --
# "too_many_words"/"trailing_punctuation"/"prose_continuation"/"list_item"
# are deliberately excluded from this set: those mean "reads like ordinary
# body prose that was mis-promoted," and demoting-and-dropping a heading on
# that evidence alone risks discarding real content, which this module
# never does.
_JUNK_HEADING_REASONS = frozenset({
    "code_or_algorithm", "equation_or_matrix", "no_letters",
    "punctuation_only", "ocr_fragment", "figure_or_table_caption",
})
# The title (H1) has its own validity rules rather than the general
# body-heading classifier, exactly as `validation.py`'s own
# `_validate_heading_quality` already carves out ("a short/minimal title is
# not the same failure mode as a code/equation/fragment line getting
# promoted to a body heading, so it's exempt from this particular scan") --
# a real single- or two-word title must not be swept up by reasons calibrated
# for body headings (e.g. "ocr_fragment"'s all-words-too-short check). Only
# the reasons that mean the heading isn't prose at all -- no letters
# whatsoever (a bare arXiv ID/DOI/page number), or no alphanumeric content --
# apply to a level-1 heading.
_JUNK_TITLE_REASONS = frozenset({"no_letters", "punctuation_only"})


def _is_junk_heading(text: str, level: Optional[int] = None) -> bool:
    reason = heading_rejection_reason(text or "")
    if reason is None:
        return False
    if level == 1:
        return reason in _JUNK_TITLE_REASONS
    return reason in _JUNK_HEADING_REASONS


# The canonical arXiv preprint stamp line PDF renderers glue onto page 1
# ("1804.08137v2 [stat.CO] 30 Nov 2018 arXiv ..."): an arXiv-ID-shaped token
# (`_ARXIV_ID_RE`'s shape, but matched anywhere in the line rather than
# requiring the whole line to be just the ID) immediately followed by a
# bracketed category tag, or the literal word "arXiv" near one -- a
# universal arXiv formatting convention (spec EXCLUDE "arXiv identifier and
# header"), not wording tied to any one paper. Length-gated the same way
# `front_matter.py`'s own front-matter-line regexes are (e.g. `_COPYRIGHT_RE`
# at <=40 words) so an ordinary long paragraph that happens to cite an arXiv
# ID mid-sentence is never swept up by this.
_ARXIV_STAMP_RE = re.compile(
    r"\d{4}\.\d{4,5}(?:v\d+)?\s*\[[a-zA-Z.\-]+\]"
    r"|\barxiv\b.{0,20}\d{4}\.\d{4,5}\b"
    r"|\d{4}\.\d{4,5}\b.{0,20}\barxiv\b",
    re.IGNORECASE,
)


def _is_arxiv_stamp_line(text: str) -> bool:
    words = (text or "").split()
    return bool(words) and len(words) <= 20 and bool(_ARXIV_STAMP_RE.search(text))


# A reserved section-label word ("Abstract", "Keywords", "Index Terms")
# appearing near the start of an otherwise-untyped paragraph in the
# front-matter zone -- the same closed, generic label vocabulary
# `validation.py`'s `_TITLE_BLOCKING_LABELS` and `front_matter.py`'s
# `_KEYWORDS_RE` already treat as reserved, extended here to catch the
# label showing up *mid-line* (e.g. an affiliation clause glued directly
# onto "... Abstract. This article ..." with no paragraph break between
# them, a column-merge extraction artifact) rather than only when it opens
# the line. Checked against the first few words only, so a body paragraph
# later in the document that happens to use "abstract" as an ordinary
# English word is never affected (the front-matter-zone caller only ever
# applies this before real body content has been seen at all).
_RESERVED_LABEL_RE = re.compile(r"\b(abstract|keywords|index terms)\b\s*[:.]?", re.IGNORECASE)


def _starts_with_reserved_label(text: str, lookahead_words: int = 10) -> bool:
    words = (text or "").split()
    return bool(_RESERVED_LABEL_RE.search(" ".join(words[:lookahead_words])))


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _front_matter_zone(blocks: List[Block]) -> Tuple[set, List[str]]:
    """Scans the document's leading prefix -- the region ordinary academic
    front matter (title, authors, affiliations, contact info, arXiv/
    publication stamp, abstract, keywords) occupies before real body content
    begins -- and returns (a) the `id()`s of every block in that prefix that
    isn't already excluded by type or already destined to become the
    resolved title, and (b) the short un-typed text fragments found there
    that look like pieces of a wrapped title, in document order.

    Exists because `EXCLUDED_BLOCK_TYPES` can only drop front matter the
    extractor successfully classified into a dedicated block type -- when a
    line lands as a plain `ParagraphBlock`/`UnknownBlock` instead (a glued
    affiliation+abstract run, an arXiv stamp line, a title that never got
    reassembled from its wrapped source lines), that per-type filter has no
    signal to act on at all. This reuses the same already-existing
    classifiers as the rest of this module (`heading_rejection_reason`,
    `numbering_tuple`) plus two cheap, generic text shapes (a bare comma, an
    email address) to recognize that content positionally instead, the same
    "re-run an existing classifier" principle `_is_shell_heading` already
    established. `classify_front_matter_line` is deliberately *not* used
    here -- its `_looks_like_author_line` shape check (2-4 Title-Case tokens
    per comma/and-separated group) matches an ordinary short title fragment
    ("In Uncertain Domains") exactly as readily as a real author line,
    which would corrupt title reconstruction; the comma/email checks below
    are narrower and cover the same real-world cases without that risk.

    The document's very first block, when it's a `HeadingBlock` that isn't
    junk, does *not* end the zone by itself -- a title/subtitle can survive
    extraction as an early, standalone heading while the rest of the title
    (or the whole thing) lands as separate paragraph text further down, past
    an arXiv stamp or other front matter (spec "reconstruct the title from
    adjacent title regions"). It's excluded from the body only in
    retrospect, once the scan actually finds separate title-region text
    elsewhere in the zone (`resolve_main_text_title` independently decides
    whether to use it as a subtitle) -- if nothing else in the zone turns
    out to be title-shaped, it's left alone as ordinary kept content (the
    common case: it simply *is* the title, standing alone).

    Stops at the first other sign real body content has begun: a
    `numbering_tuple` match (a genuine numbered section, e.g.
    "1 Introduction", regardless of whether it was promoted to a
    `HeadingBlock`), a later non-junk `HeadingBlock`, or a substantial
    (>=20-word) paragraph that doesn't match any front-matter/label/
    arXiv-stamp signal -- from that point on, `build_main_text_document`'s
    ordinary rules apply unchanged."""
    excluded_ids: set = set()
    title_parts: List[str] = []
    leading_heading = None
    # True once *something* in the zone gave strong, unambiguous evidence
    # of front matter (an arXiv stamp, a reserved-label line, or an
    # already-typed front-matter block) -- required before a short,
    # otherwise-unclassified paragraph is trusted as a title-continuation
    # fragment rather than the start of ordinary body prose. Without this
    # gate, "H1 title" immediately followed by a short, perfectly ordinary
    # first sentence would misread that sentence as more of the title --
    # the real documents this module targets always have an arXiv stamp or
    # a typed front-matter block interposed between a split title/subtitle
    # and its continuation, so requiring that evidence first costs nothing
    # on the documents this exists for, while protecting the far more
    # common "title, then body" shape from being swept up.
    saw_strong_signal = False
    for i, b in enumerate(blocks):
        text = (getattr(b, "text", None) or "").strip()

        if isinstance(b, HeadingBlock):
            if numbering_tuple(text) is not None:
                break  # a real numbered section heading: zone ends here
            if _is_junk_heading(text, getattr(b, "level", None)):
                excluded_ids.add(id(b))
                continue
            if i == 0:
                leading_heading = b  # tentatively kept; see docstring
                continue
            break  # a later, real, non-junk heading: zone ends here

        if isinstance(b, EXCLUDED_BLOCK_TYPES):
            excluded_ids.add(id(b))  # already excluded by type; stays in-zone
            saw_strong_signal = True
            continue

        if isinstance(b, (ParagraphBlock, UnknownBlock)):
            if not text:
                excluded_ids.add(id(b))
                continue
            if numbering_tuple(text) is not None:
                break  # e.g. "1 Introduction" surfaced as a plain block
            if _is_arxiv_stamp_line(text):
                excluded_ids.add(id(b))
                saw_strong_signal = True
                continue
            if _starts_with_reserved_label(text):
                excluded_ids.add(id(b))
                saw_strong_signal = True
                continue
            words = text.split()
            if len(words) < 20 and saw_strong_signal:
                # A short leftover fragment with no comma and no email is
                # treated as a candidate piece of a wrapped title; one that
                # has either is far more likely author/contact-list debris
                # (a bare comma essentially never appears in a title line,
                # author lists almost always have one; an email is never
                # part of a title) -- shape, not wording.
                if "," not in text and not _EMAIL_RE.search(text):
                    title_parts.append(text)
                excluded_ids.add(id(b))
                continue
            break  # substantial, or short with no prior front-matter evidence: real body begins

        # Any other block type reached while still in the zone (list/table/
        # code/image/...) means real body content (or an already-typed
        # figure/table this module handles elsewhere) has begun.
        break

    if title_parts and leading_heading is not None:
        # Separate title-region text was found elsewhere in the zone, so
        # the leading heading was a subtitle fragment, not independent body
        # content -- fold it out of the body too (`resolve_main_text_title`
        # is responsible for actually combining it into `document_title`).
        excluded_ids.add(id(leading_heading))

    return excluded_ids, title_parts


def _is_mislabeled_abstract_heading(blocks: List[Block], idx: int) -> bool:
    """True when `blocks[idx]` is a non-title `HeadingBlock` that is itself
    author/affiliation/contact-list shaped (i.e. would have classified as
    front matter had it landed as an ordinary paragraph instead of being
    promoted to a heading by the extractor's font/geometry heuristics --
    reuses `front_matter.classify_front_matter_line`'s exact generic name-
    list shape check, not a new heuristic) AND is immediately followed by
    content that itself opens with the reserved "Abstract"/"Summary"/
    "Keywords" label.

    Real-world shape this catches: a PDF renders its author byline in a
    large/bold font just like a section heading, so the extractor promotes
    "Marco Scutari - Claudia Vitolo - Allan Tucker" to an H2 `HeadingBlock`
    -- with the paper's actual Abstract paragraph sitting directly beneath
    it. `is_abstract_heading` alone can't catch this: it matches heading
    *text* against "Abstract"/"Summary", and this heading's text is an
    author list, not that word. Without this check, the abstract prose
    goes straight into MAIN_TEXT under a heading that reads as author
    names -- exactly the kind of publication-front-matter leak spec
    section 3B's EXCLUDE list rules out. Scoped narrowly (author-shaped
    heading textually followed by a reserved-label paragraph) so an
    ordinary body heading is never at risk of being misread this way."""
    b = blocks[idx]
    if not (isinstance(b, HeadingBlock) and idx > 0):
        return False
    if classify_front_matter_line(b.text or "") != "author":
        return False
    # Scan forward past short, front-matter-shaped lines that commonly sit
    # between an author byline and the Abstract paragraph itself (e.g. an
    # editorial "Received: .../ Accepted: ..." timeline line) -- stopping
    # at the next heading (section ends without finding the label) or the
    # first substantial (>=8-word) paragraph that ISN'T itself front-
    # matter-shaped (real body content has begun; nothing this far in
    # counts as still being "immediately after" the heading).
    j = idx + 1
    while j < len(blocks):
        nxt = blocks[j]
        if isinstance(nxt, HeadingBlock):
            return False
        text = (getattr(nxt, "text", None) or "").strip()
        if text and _starts_with_reserved_label(text):
            return True
        if text and len(text.split()) >= 8 and not classify_front_matter_line(text):
            return False
        j += 1
    return False


# A numbered bibliography entry -- "[N] Author, Author, and Author. YYYY.
# Title..." -- the near-universal ACM/IEEE numbered-citation shape. This is
# a MAIN_TEXT exclusion signal independent of `is_references_heading`, for
# a document whose References section lost its heading entirely during
# extraction (a two-column PDF/OCR layout that flattened the whole
# bibliography into plain paragraphs indistinguishable-by-heading from the
# body) -- this module can only act on already-classified blocks, and the
# heading-stack scan in `build_main_text_document` has nothing to key off
# when no References `HeadingBlock` was ever produced. Requires the
# bracketed-number prefix *and* a publication year within the same entry --
# a real citation's "Author Name(s). YYYY." shape -- not a lone "starts
# with [N]" match, so an ordinary body paragraph that happens to open with
# a bracketed citation marker is not swept up by this alone.
_REFERENCE_ENTRY_START_RE = re.compile(r"^\s*\[\d{1,3}\]\s+\S")
_PUBLICATION_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _looks_like_reference_list_entry(text: str) -> bool:
    if not _REFERENCE_ENTRY_START_RE.match(text or ""):
        return False
    # The year in a real citation follows the author list closely ("...
    # Tucker. 2019. Title..."); capping the search window keeps a body
    # paragraph that opens with a bracketed reference marker and mentions
    # an unrelated year much later from being caught by coincidence. Wide
    # enough for a real (if long) author list -- some venues credit 15-20
    # co-authors before the year -- without growing so large it risks
    # matching a year that's actually part of unrelated later content.
    return bool(_PUBLICATION_YEAR_RE.search(text[:400]))


def build_main_text_document(document: Document) -> Tuple[Document, MainTextFilterStats]:
    """Returns a new `Document` sharing `document`'s metadata and a filtered
    subset of its `.blocks` (same `Block` objects, not copies -- block_id
    and page provenance are untouched, `.order` is left at each block's
    original position rather than renumbered, since nothing downstream of
    this view keys off contiguous ordering, only list order, and leaving it
    alone means this can never corrupt a shared block object read
    elsewhere). Never mutates `document.blocks` itself -- a FULL_FIDELITY
    call on the same extraction result sees the original, unfiltered list."""
    stats = MainTextFilterStats(total_source_blocks=len(document.blocks))
    blocks = document.blocks

    # Section membership (References/Bibliography, Table of Contents) is a
    # heading-path property, computed once over the *original* block order
    # -- a heading's own body-content exclusion (shell-heading, below)
    # never changes which blocks are inside one of these two named
    # sections.
    in_references: set = set()
    in_toc: set = set()
    in_abstract: set = set()
    in_acknowledgements: set = set()
    # Computed up front (needs index-based lookahead, same as
    # `shell_heading_ids` below) so the stack loop can treat an author-
    # byline-shaped heading immediately followed by an "Abstract"-labelled
    # paragraph the same way it already treats a literal "## Abstract"
    # heading -- see `_is_mislabeled_abstract_heading`'s docstring.
    mislabeled_abstract_heading_ids = {
        id(blocks[i]) for i in range(len(blocks)) if _is_mislabeled_abstract_heading(blocks, i)
    }
    stack: List[HeadingBlock] = []
    current_refs = False
    current_toc = False
    current_abstract = False
    current_ack = False
    for b in blocks:
        if isinstance(b, HeadingBlock):
            while stack and stack[-1].level >= b.level:
                stack.pop()
            stack.append(b)
            current_refs = any(is_references_heading(h.text or "") for h in stack)
            current_toc = any(is_toc_heading(h.text or "") for h in stack)
            current_abstract = any(
                is_abstract_heading(h.text or "") or id(h) in mislabeled_abstract_heading_ids for h in stack
            )
            current_ack = any(is_acknowledgements_heading(h.text or "") for h in stack)
        if current_refs:
            in_references.add(id(b))
        if current_toc:
            in_toc.add(id(b))
        if current_abstract:
            in_abstract.add(id(b))
        if current_ack:
            in_acknowledgements.add(id(b))

    first_heading_idx = next((i for i, b in enumerate(blocks) if isinstance(b, HeadingBlock)), None)
    shell_heading_ids = {
        id(b) for i, b in enumerate(blocks)
        if isinstance(b, HeadingBlock) and _is_shell_heading(blocks, i, first_heading_idx)
    }
    junk_heading_ids = {
        id(b) for b in blocks
        if isinstance(b, HeadingBlock) and _is_junk_heading(b.text or "", b.level)
    }
    zone_excluded_ids, title_parts = _front_matter_zone(blocks)
    stats.title_region_text = " ".join(title_parts).strip() or None

    kept: List[Block] = []
    for b in blocks:
        if id(b) in in_references:
            stats.excluded_references_blocks += 1
            continue
        if isinstance(b, (ParagraphBlock, UnknownBlock)) and _looks_like_reference_list_entry(
            getattr(b, "text", None) or ""
        ):
            # Same exclusion outcome as `in_references` above, just reached
            # by entry shape instead of heading membership -- see
            # `_looks_like_reference_list_entry`'s docstring.
            stats.excluded_references_blocks += 1
            continue
        if id(b) in in_toc:
            stats.excluded_toc_blocks += 1
            continue
        if id(b) in in_abstract:
            stats.excluded_abstract_blocks += 1
            continue
        if id(b) in in_acknowledgements:
            stats.excluded_acknowledgements_blocks += 1
            continue
        if id(b) in shell_heading_ids:
            stats.excluded_shell_heading_blocks += 1
            continue
        if id(b) in junk_heading_ids:
            stats.excluded_junk_heading_blocks += 1
            continue
        if isinstance(b, EXCLUDED_BLOCK_TYPES):
            name = type(b).__name__
            stats.excluded_by_type[name] = stats.excluded_by_type.get(name, 0) + 1
            continue
        if id(b) in zone_excluded_ids:
            stats.excluded_front_matter_zone_blocks += 1
            continue
        kept.append(b)

    stats.included_blocks = len(kept)
    filtered = Document(metadata=document.metadata, blocks=kept)
    return filtered, stats


# A bare publication identifier -- an arXiv accession number (with or
# without its "vN" revision suffix) or a DOI -- occasionally ends up as a
# PDF's embedded /Title metadata field on its own, with no document text
# ever containing it verbatim (so `_title_matches_excluded_content`'s
# exact-block-text match can't catch it): a browser "Print to PDF" of an
# arXiv page, for instance, stamps the arXiv ID into /Title as the source
# tab's title, not the paper's. This is unambiguously "arXiv metadata" /
# "DOI ... metadata" (spec EXCLUDE list), never a title, regardless of
# which publisher's pipeline produced it -- generic identifier syntax, not
# a match against any one paper's wording.
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_DOI_RE = re.compile(r"^(doi:\s*)?10\.\d{4,9}/\S+$", re.IGNORECASE)


def _is_bare_publication_identifier(title: str) -> bool:
    norm = title.strip()
    return bool(_ARXIV_ID_RE.match(norm) or _DOI_RE.match(norm))


def _title_matches_excluded_content(title: str, blocks: List[Block]) -> bool:
    """True if `title` is verbatim the text of a block that MAIN_TEXT would
    never treat as body content -- a caption, a figure label, publication
    metadata, a References/TOC heading, or a running-header/page-furniture
    line already classified as such. Guards `resolve_main_text_title`
    against `document.metadata.title` itself having been resolved (by the
    extractor, upstream of this module) from one of those regions."""
    norm = " ".join(title.strip().lower().split())
    if not norm:
        return False
    for b in blocks:
        text = (getattr(b, "text", None) or "").strip()
        if not text:
            continue
        text_norm = " ".join(text.lower().split())
        if text_norm != norm:
            continue
        if isinstance(b, EXCLUDED_BLOCK_TYPES):
            return True
        if isinstance(b, HeadingBlock) and (is_references_heading(text) or is_toc_heading(text)):
            return True
    return False


def _leading_subtitle_candidate(document: Document) -> Optional[str]:
    """The document's very first block, *only* when it's a `HeadingBlock`
    that is itself a real title/subtitle fragment, not junk (`_is_junk_heading`
    -- e.g. a bare arXiv ID promoted straight to H1; title-level junk reasons
    only, same as any other H1 candidate). Used only to pair with a
    separately reconstructed main title (`title_region_text`) as "{main}:
    {subtitle}" -- deliberately restricted to position 0 only (never "the
    next heading found," which risks picking up an unrelated body subsection
    deep in the document instead of a real subtitle)."""
    blocks = document.blocks
    if not blocks or not isinstance(blocks[0], HeadingBlock):
        return None
    text = (blocks[0].text or "").strip()
    if not text or _is_junk_heading(text, blocks[0].level):
        return None
    return text


def resolve_main_text_title(
    document: Document, filtered_blocks: List[Block], title_region_text: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """MAIN_TEXT title resolution (spec Phase 5, and the main-title/subtitle
    extension): only title-eligible textual content may become the resolved
    title, in priority order:

    1. `document.metadata.title` (PDF metadata or first-page visual-title
       resolution) when present, not itself shaped like excluded content,
       and not a bare publication identifier (arXiv ID, DOI --
       `_is_bare_publication_identifier`) that ended up in /Title metadata
       with no document text to match against verbatim. This is what lets a
       document whose body H1 structure is broken (two H1s, a figure label
       promoted to H1, ...) still resolve to its real title under MAIN_TEXT
       without needing that body structure repaired first.

    2. `title_region_text` (`_front_matter_zone`'s reconstruction of a
       wrapped/never-typed title from the document's front-matter zone),
       combined with the document's leading heading as a subtitle when that
       heading is itself a genuine, distinct title fragment
       (`_leading_subtitle_candidate`) rather than junk or front matter --
       covers a document whose real title never got assembled into one
       block at all (spec "reconstruct the title from adjacent title
       regions"; "support a main title plus subtitle").

    3. `title_region_text` alone, when no separate subtitle candidate exists.

    4. The first `HeadingBlock` remaining in the *filtered* view --
       graphical labels, captions, running headers, TOC/References/Abstract/
       Acknowledgements entries, publication metadata, and junk/front-matter-
       shaped headings have already been removed from that view by
       `build_main_text_document`, so none of them can ever reach this
       fallback."""
    meta_title = (document.metadata.title or "").strip()
    if meta_title and not _title_matches_excluded_content(meta_title, document.blocks) \
            and not _is_bare_publication_identifier(meta_title):
        return meta_title, "metadata"

    region = (title_region_text or "").strip()
    if region:
        subtitle = _leading_subtitle_candidate(document)
        if subtitle and subtitle.strip().lower() != region.lower():
            return f"{region}: {subtitle}", "reconstructed_title_and_subtitle"
        return region, "reconstructed_title_region"

    for b in filtered_blocks:
        if isinstance(b, HeadingBlock):
            text = (b.text or "").strip()
            if text:
                return text, "first_body_heading"
    return None, "unresolved"


def apply_resolved_title(active_document: Document, title: Optional[str]) -> None:
    """Ensures `title` (`resolve_main_text_title`'s output, or the pipeline's
    filename-derived fallback when even that resolved to nothing) is what
    the Markdown renderer, the chunker, and every other downstream reader
    of `active_document` actually see as the document's opening H1 --
    title resolution and what gets rendered/chunked must never drift apart
    (spec: "the resulting Markdown must begin" with the resolved title).
    Without this, `render_markdown(active_document)` -- which only ever
    walks `active_document.blocks` in order, with no special handling of
    "the title" at all -- opens with whatever `HeadingBlock` happened to
    survive filtering first, which is only the resolved title by
    coincidence: never when it came from `document.metadata.title` or from
    `_front_matter_zone`'s reconstruction, and not even in the
    "first_body_heading" fallback case unless that heading was also the
    very first surviving block.

    Mutates `active_document.blocks` in place (this is already the
    MAIN_TEXT-filtered, per-call view `build_main_text_document` built --
    never the original `document` a FULL_FIDELITY call would still see, and
    never called for FULL_FIDELITY at all). Removes any other level-1
    `HeadingBlock` whose text matches `title` verbatim first, so the title
    is never rendered twice -- once as the synthesized opener, once again
    as an ordinary body heading further down -- then prepends a fresh one
    if the document doesn't already open with it, following the exact same
    "synthesize a `HeadingBlock` for a resolved-but-missing title" pattern
    `normalize.py`'s `_enforce_single_first_h1` already established for
    FULL_FIDELITY (`next_block_id()`, `level=1`, `text`/`normalized` both
    set) -- not a new mechanism. A no-op when `title` is falsy (nothing was
    resolved at all; MAIN_TEXT readiness already blocks that case
    separately, so there is nothing to render as a title-shaped no-op)."""
    if not title:
        return
    norm = " ".join(title.strip().lower().split())
    blocks = active_document.blocks

    def _is_same_title(b: Block) -> bool:
        return (
            isinstance(b, HeadingBlock) and b.level == 1
            and " ".join((b.text or "").strip().lower().split()) == norm
        )

    if blocks and _is_same_title(blocks[0]):
        return  # already exactly right
    kept = [b for b in blocks if not _is_same_title(b)]
    title_block = HeadingBlock(
        block_id=active_document.next_block_id(), order=-1,
        source_format=active_document.metadata.source_format, level=1, text=title, normalized=title,
    )
    active_document.blocks = [title_block] + kept
