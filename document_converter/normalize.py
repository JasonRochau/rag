"""
Deterministic, block-level structural normalization (spec section 7).

Whitespace correctness (blank-line handling, ATX spacing, one-blank-line
separation) is guaranteed by construction in `md_parser.py` (parse) and
`render.py` (render) -- see the module docstring in render.py for why that
is a deliberate design choice rather than an oversight. What is left for
this module is the part that genuinely requires reasoning about the
document as a whole: heading-level nesting and the single/first-H1 rule.
Every repair made here is recorded as a FIXED finding so it shows up in the
audit log required by the spec.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from .model import (
    Document, HeadingBlock, ParagraphBlock, UnknownBlock, WarningBlock, CaptionBlock,
    FRONT_MATTER_BLOCK_TYPES, AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock,
    PublicationMetadataBlock, CopyrightNoticeBlock, PermissionNoticeBlock, ConferenceMetadataBlock,
)
from .findings import FindingLog
from .extractors import looks_like_caption_text
from .front_matter import classify_front_matter_line

# Subject-level words that name a whole field/topic rather than *this*
# document within it (spec section 9's "broad subject label" penalty).
# Deliberately generic (not tuned to any one fixture) -- a document whose
# only title is one of these carries essentially no provenance value on its
# own ("Algorithms" could be any of a thousand documents).
_GENERIC_SUBJECT_LABELS = {
    "algorithms", "algorithm", "introduction", "overview", "summary", "notes",
    "chapter", "basics", "fundamentals", "review", "programming", "exercises",
    "abstract", "contents", "appendix", "index", "glossary", "notes",
}
_MINOR_TITLE_WORDS = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "or", "nor", "vs", "with", "by"}
_ACRONYM_RE = re.compile(r"^[A-Z0-9]+$")


def _slug_title_from_path(source_path: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return clean_filename_title(stem)


def clean_filename_title(stem: str) -> str:
    """Turns a source filename stem into a readable candidate title.

    A ' - ' (hyphen surrounded by whitespace) is treated as a subtitle
    divider, not an ordinary word separator, and kept as an em dash; a
    "Word123" filename artifact (letters glued directly to digits) is split
    into "Word 123"; acronym/code-like tokens ("CSC506") are preserved
    verbatim instead of being mangled by naive capitalization; minor words
    (a/of/and/...) are lowercased except at the start or end.
    """
    stem = re.sub(r"\s+-\s+", " — ", stem)
    stem = stem.replace("_", " ")
    # A downloaded/exported filename commonly substitutes a hyphen for
    # every space in the title ("Paper-Title-Here.pdf"), the same role
    # underscore plays above -- but a hyphen directly touching a digit on
    # either side is far more often a real compound (a page/year range,
    # "COVID-19") than a word separator, so only letter-to-letter hyphens
    # are converted, never a digit-adjacent one.
    stem = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", stem)
    stem = re.sub(r"([a-z])(\d)", r"\1 \2", stem)
    words = [w for w in stem.split(" ") if w]
    out = []
    for i, w in enumerate(words):
        if w == "—":
            out.append(w)
        elif _ACRONYM_RE.match(w) or any(ch.isdigit() for ch in w):
            out.append(w)
        elif w.lower() in _MINOR_TITLE_WORDS and 0 < i < len(words) - 1:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out) or "Untitled Document"


def title_distinctiveness_score(title: str) -> Tuple[float, str]:
    """Heuristic 0..1 score for how distinctive/specific a candidate title
    is, plus a short reason string. Penalizes a single generic word, a
    broad subject-label word, and a short title with no distinguishing
    number or multi-word specificity at all (spec section 9)."""
    words = [w for w in re.split(r"\s+", title.strip()) if w]
    if not words:
        return 0.0, "empty title"
    score = 1.0
    reasons = []
    if len(words) == 1:
        score -= 0.5
        reasons.append("one generic word")
    if title.strip().lower().rstrip(".:") in _GENERIC_SUBJECT_LABELS:
        score -= 0.4
        reasons.append("broad subject label")
    if len(words) <= 2 and not any(any(ch.isdigit() for ch in w) for w in words):
        score -= 0.1
        reasons.append("short with no distinguishing number")
    score = max(0.0, min(1.0, score))
    return score, ("; ".join(reasons) if reasons else "distinctive")


def _filename_title_similarity(candidate_title: str, filename_title: str) -> float:
    """Fraction of `candidate_title`'s words also present in
    `filename_title` -- 1.0 means the candidate is essentially a subset of
    the (more descriptive) filename-derived title."""
    cand_words = {w.lower() for w in re.split(r"\s+", candidate_title) if w}
    fn_words = {w.lower() for w in re.split(r"\s+", filename_title) if w}
    if not cand_words:
        return 0.0
    return len(cand_words & fn_words) / len(cand_words)


def normalize_document(document: Document) -> FindingLog:
    """Repairs `document` in place. Returns the audit log of repairs made."""
    log = FindingLog()
    _reconstruct_split_title(document, log)
    _enforce_single_first_h1(document, log)
    _repair_heading_nesting(document, log)
    _reconstruct_leading_front_matter(document, log)
    _normalize_figure_captions(document, log)
    return log


# Words that can never end a grammatically complete title/subtitle on their
# own -- articles, coordinating conjunctions, prepositions, and possessive/
# demonstrative determiners all obligatorily take an object that must
# follow. A heading-shaped line ending in one of these is mid-phrase, not a
# short-but-complete title (spec section 3A). Deliberately the same kind of
# closed, part-of-speech-based list `_MINOR_TITLE_WORDS` already uses for
# title-casing, extended with possessives/demonstratives since those are the
# words that actually show up as truncation points ("... and Their",
# "... of the").
_DANGLING_TITLE_WORDS = _MINOR_TITLE_WORDS | {
    "their", "its", "his", "her", "our", "your", "my",
    "this", "that", "these", "those", "as", "at", "into", "onto", "from",
}


def _ends_with_dangling_word(text: str) -> bool:
    words = re.split(r"\s+", (text or "").strip())
    if not words or not words[-1]:
        return False
    last = words[-1].strip(".,;:!?\"'()[]").lower()
    return last in _DANGLING_TITLE_WORDS


def _reconstruct_split_title(document: Document, log: FindingLog, max_merges: int = 3) -> None:
    """A title wrapped/OCR-split across two adjacent title-shaped regions
    (e.g. an H1 "Regression Analyses and Their" immediately followed, once
    any already-grouped front matter in between is skipped, by an H2
    "Particularities in Observational Studies") is not two structural
    elements -- it is one title whose first region ends mid-phrase. This is
    a grammatical-completeness test (`_ends_with_dangling_word`), not a
    length or position heuristic, precisely so it does *not* fire on a
    genuinely complete short title immediately followed by a legitimate
    bodyless subtitle (spec section 5's "must not merge a legitimate
    subtitle" requirement) -- a real subtitle heading doesn't end on a
    dangling article/conjunction/preposition/possessive.

    Runs before `_enforce_single_first_h1` so title-move/front-matter
    repairs downstream see the complete title, not a truncated fragment of
    it. Skips over already-classified front-matter blocks when looking for
    the continuation (front-matter extraction can legitimately sit between
    a multi-line title's regions if the source's raw reading order
    interleaved a sidebar/masthead column with the title column), but never
    crosses onto a different page or past a non-heading, non-front-matter
    block, so it can never reach into unrelated body content."""
    for _ in range(max_merges):
        headings = [b for b in document.blocks if isinstance(b, HeadingBlock)]
        h1s = [h for h in headings if h.level == 1]
        if not h1s:
            return
        primary = h1s[0]
        if not _ends_with_dangling_word(primary.text):
            return
        idx = document.blocks.index(primary)
        j = idx + 1
        while j < len(document.blocks) and isinstance(document.blocks[j], FRONT_MATTER_BLOCK_TYPES):
            j += 1
        if j >= len(document.blocks):
            return
        nxt = document.blocks[j]
        if not isinstance(nxt, HeadingBlock):
            return
        if primary.page is not None and nxt.page is not None and primary.page != nxt.page:
            return
        merged_text = f"{primary.text.rstrip()} {nxt.text.lstrip()}".strip()
        old_text = primary.text
        primary.text = merged_text
        primary.normalized = merged_text
        document.blocks.pop(j)
        _renumber(document)
        log.fixed(
            "title_continuation_merged",
            f"Merged title continuation {nxt.text!r} (source region {nxt.block_id}, "
            f"page {nxt.page}) into H1 {old_text!r}, split across adjacent title-shaped "
            f"regions: now {merged_text!r}",
            block_id=primary.block_id,
        )


_FRONT_MATTER_KIND_CTORS = {
    "author": AuthorBlock, "affiliation": AffiliationBlock, "contact": ContactBlock,
    "keywords": KeywordBlock, "publication_metadata": PublicationMetadataBlock,
    "copyright": CopyrightNoticeBlock, "permission": PermissionNoticeBlock,
    "conference": ConferenceMetadataBlock,
}


def _reconstruct_leading_front_matter(document: Document, log: FindingLog) -> None:
    """Parser-neutral backstop for author/affiliation/contact/conference
    text sitting in the title-adjacent run that extraction-time
    classification missed -- most commonly a line OCR (or a parser with no
    front-matter classification of its own at all, e.g. the Docling
    adapter) glued multiple fields onto one physical line
    (`classify_front_matter_line`'s compound-author-line shape) or a
    running conference/venue line, both of which a purely per-*raw*-line
    classifier run before paragraph merging can miss.

    Deliberately narrow: scans forward from the title only through the
    *contiguous* run immediately following it, on the title's own page,
    stopping at the first block that is neither already front-matter-typed
    nor independently front-matter-shaped -- the same "one contiguous group
    right after the title" boundary `_validate_front_matter_placement`
    itself checks for, so this can never reach into the body and relabel
    an unrelated paragraph."""
    h1_idx = next((i for i, b in enumerate(document.blocks) if isinstance(b, HeadingBlock) and b.level == 1), None)
    if h1_idx is None:
        return
    title_page = document.blocks[h1_idx].page
    i = h1_idx + 1
    while i < len(document.blocks):
        b = document.blocks[i]
        if isinstance(b, FRONT_MATTER_BLOCK_TYPES):
            i += 1
            continue
        if not isinstance(b, ParagraphBlock) or (title_page is not None and b.page != title_page):
            break
        text = (b.text or b.normalized or "").strip()
        kind = classify_front_matter_line(text) if text else None
        if kind is None:
            break
        ctor = _FRONT_MATTER_KIND_CTORS[kind]
        typed = ctor(
            block_id=b.block_id, order=b.order, source_format=b.source_format,
            raw=b.raw, page=b.page, text=text, normalized=b.normalized or text,
        )
        typed.style = dict(b.style)
        document.blocks[i] = typed
        log.fixed(
            "front_matter_reconstructed",
            f"Reclassified title-adjacent {type(b).__name__} as {kind} front matter "
            f"(source region {b.block_id}, page {b.page}): {text!r}",
            block_id=b.block_id, page=b.page,
        )
        i += 1


_CAPTION_ELIGIBLE_TYPES = (ParagraphBlock, UnknownBlock, WarningBlock)


def _normalize_figure_captions(document: Document, log: FindingLog) -> None:
    """A figure/table caption line ("Figure 3 ...", "Table 1: ...") must
    always end up a `CaptionBlock`, regardless of source format or parser --
    never left as ordinary body prose, and (per `heading_rejection_reason`/
    `demote_false_headings`, enforced elsewhere) never promoted to a
    heading either. This is the parser-neutral backstop for a caption line
    that was never itself a *heading candidate* in the first place (e.g. it
    never had the isolated/font-size shape `pdf_extractor.py`'s OCR scorer
    requires before it even applies `classify_ocr_nonheading_kind`'s
    caption/figure-label reclassification) -- an ordinary-looking body
    paragraph whose text is nonetheless caption-shaped still needs to be
    pulled out, so a run of "Figure 2 ...", "Figure 3 ...", "Figure 4 ..."
    is classified consistently regardless of which one happened to look
    heading-sized to the extractor."""
    for i, b in enumerate(document.blocks):
        if not isinstance(b, _CAPTION_ELIGIBLE_TYPES):
            continue
        text = (getattr(b, "text", None) or "").strip()
        if not text or not looks_like_caption_text(text):
            continue
        caption = CaptionBlock(
            block_id=b.block_id, order=b.order, source_format=b.source_format,
            raw=b.raw, page=b.page, text=text, normalized=b.normalized or text,
        )
        caption.style = dict(b.style)
        document.blocks[i] = caption
        log.fixed(
            "figure_caption_normalized",
            f"Reclassified {type(b).__name__} {text!r} as a caption block "
            f"(source region {b.block_id}, page {b.page})",
            block_id=b.block_id, page=b.page,
        )


def _headings(document: Document):
    return [b for b in document.blocks if isinstance(b, HeadingBlock)]


def _enforce_single_first_h1(document: Document, log: FindingLog) -> None:
    headings = _headings(document)
    h1s = [h for h in headings if h.level == 1]

    if len(h1s) == 0:
        title = document.metadata.title or _slug_title_from_path(document.metadata.source_path)
        new_h1 = HeadingBlock(
            block_id=document.next_block_id(), order=0, source_format=document.metadata.source_format,
            level=1, text=title, normalized=title,
        )
        document.blocks.insert(0, new_h1)
        _renumber(document)
        log.fixed("synthesized_h1", f"No H1 found; synthesized title heading {title!r}", block_id=new_h1.block_id)
        h1s = [new_h1]

    elif len(h1s) > 1:
        primary = h1s[0]
        for extra in h1s[1:]:
            extra.level = 2
            log.fixed("demoted_extra_h1", f"Demoted extra H1 {extra.text!r} to H2", block_id=extra.block_id)

    primary = h1s[0]
    idx = document.blocks.index(primary)
    if idx != 0:
        if idx <= 2:
            # Carry along any already-contiguous front-matter run immediately
            # following the title's *current* position, so a title/front-matter
            # adjacency that extraction-time grouping already got right (spec
            # sections 5/6/14, enforced by validation.py's
            # _validate_front_matter_placement) survives the move instead of
            # being separated from the title by whatever used to precede it
            # (e.g. stray page-furniture/continuation text ahead of the title
            # region) -- a later repair must never invalidate an earlier one's
            # already-correct output.
            end = idx + 1
            while end < len(document.blocks) and isinstance(document.blocks[end], FRONT_MATTER_BLOCK_TYPES):
                end += 1
            unit = document.blocks[idx:end]
            del document.blocks[idx:end]
            document.blocks[0:0] = unit
            _renumber(document)
            moved_fm = len(unit) - 1
            msg = f"Moved H1 {primary.text!r} to the start of the document"
            if moved_fm:
                msg += (f", together with {moved_fm} contiguous front-matter block(s) that "
                        f"already immediately followed it, to preserve that adjacency")
            log.fixed("h1_moved_to_front", msg, block_id=primary.block_id)
        else:
            log.warning(
                "h1_not_first", f"H1 {primary.text!r} is not the first meaningful block (position {idx})",
                block_id=primary.block_id,
                suggested_action="Move the title heading to the start of the source document.",
            )

    _apply_title_distinctiveness_check(document, primary, log)


def _apply_title_distinctiveness_check(document: Document, primary: HeadingBlock, log: FindingLog) -> None:
    """Spec section 9: a resolved root heading that reads as a generic
    subject label ("Algorithms") rather than a title of *this* document is
    lower-provenance-value than a filename that's actually more specific
    ("CSC506 Design and Analysis of Algorithms - Chapter2.pdf") -- prefer
    the cleaned filename in that case. Never fires for a title that's
    already reasonably distinctive on its own (spec item 19)."""
    root_score, root_reason = title_distinctiveness_score(primary.text)
    filename_title = clean_filename_title(os.path.splitext(os.path.basename(document.metadata.source_path))[0])
    filename_score, _ = title_distinctiveness_score(filename_title)
    similarity = _filename_title_similarity(primary.text, filename_title)

    replaced = False
    if (
        root_score < 0.5
        and filename_score > root_score
        and similarity >= 0.8
        and len(filename_title.split()) > len(primary.text.split())
    ):
        old_title = primary.text
        primary.text = filename_title
        primary.normalized = filename_title
        replaced = True
        log.fixed(
            "generic_title_replaced_with_filename",
            f"Root heading title {old_title!r} scored low distinctiveness ({root_score:.2f}: {root_reason}) "
            f"and is a near-subset of the more descriptive filename-derived title {filename_title!r}; "
            f"replaced",
            block_id=primary.block_id,
        )

    document.metadata.extra["title_distinctiveness"] = {
        "title_distinctiveness_score": filename_score if replaced else root_score,
        "filename_title_similarity": similarity,
        "root_heading_title_score": root_score,
        "title_fallback_reason": (
            f"low-distinctiveness root heading ({root_reason}) replaced with filename-derived title"
            if replaced else "root heading title retained (distinctive enough, or filename not a better fit)"
        ),
    }


def _repair_heading_nesting(document: Document, log: FindingLog) -> None:
    max_seen = 0
    for h in _headings(document):
        if h.level > max_seen + 1:
            original = h.level
            h.level = max_seen + 1
            log.fixed(
                "skipped_heading_level",
                f"Heading {h.text!r} jumped from H{max_seen} to H{original}; repaired to H{h.level}",
                block_id=h.block_id,
            )
        max_seen = max(max_seen, h.level)


def _renumber(document: Document) -> None:
    for i, b in enumerate(document.blocks):
        b.order = i
