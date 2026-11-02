"""
Front-matter line classification (spec section 4): a pure text-shape
classifier, independent of any PDF/OCR geometry, so it can be unit-tested
without a real document and reused by any extractor. Callers are
responsible for applying positional gating on top of this (e.g. "only
trust an 'author' classification on page 1, before the abstract") --
see `extractors/pdf_extractor.py`'s front-matter pass for that.

Every pattern here is generic (email shape, copyright/permission
boilerplate phrasing, ISBN/DOI syntax, organization-name keywords,
capitalized-name-list shape) -- nothing here is specific to any one
document's title, authors, or publisher.
"""
from __future__ import annotations

import re
from typing import Optional

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PERMISSION_RE = re.compile(r"^\s*permission (to|is granted)\b", re.IGNORECASE)
# A bare "©" glyph is a frequent OCR misread of a math/pseudocode symbol
# (element-of, prime, degree, section mark) in scanned technical documents --
# a real copyright mark is essentially always immediately followed by a
# 4-digit year ("© 2020", "©2020"), so that's required for the symbol form.
# The unambiguous English phrases ("copyright", "all rights reserved") never
# need that extra corroboration.
_COPYRIGHT_RE = re.compile(r"(©\s?\d{4}|\bcopyright\b|\ball rights reserved\b)", re.IGNORECASE)
_CONFERENCE_RE = re.compile(
    r"\b(proceedings of|transactions of|conference on|symposium on|workshop on|annual meeting|"
    r"(?:spring|summer|fall|winter)\s+meeting|in\s+.*\bconference\b)\b",
    re.IGNORECASE,
)
_MONTH_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# A line ending in "<Month> <Day>[-<Day>], <Year>" -- the near-universal
# closing shape of an academic publication-venue citation ("<venue name>,
# <City>, <Country>, May 17-18, 2018") -- catches the many venue-name
# phrasings (journal/society/transactions names) the fixed phrase bank
# above can't enumerate, without matching on any one venue's own wording.
_CONFERENCE_DATE_TAIL_RE = re.compile(
    rf",\s*{_MONTH_RE}\.?\s+\d{{1,2}}\s*[-–—]?\s*\d{{0,2}}\s*,\s*\d{{4}}\s*$",
    re.IGNORECASE,
)
_ISBN_DOI_RE = re.compile(r"\bISBN\b|\bDOI\b|10\.\d{4,9}/\S+", re.IGNORECASE)
_KEYWORDS_RE = re.compile(r"^\s*(keywords|index terms)\s*[:\-]", re.IGNORECASE)
_AFFIL_KEYWORDS_RE = re.compile(
    r"\b(University|Institute|Department|Dept\.?|College|School of|Laboratory|"
    r"Lab\.?|Corporation|Inc\.|GmbH|Ltd\.?|Faculty of|Polytechnic)\b"
)
_NAME_TOKEN_RE = re.compile(r"^[A-Z][a-zA-Z.'\-]*$")
# Author-name-group separator: comma/semicolon/"and"/"&" (unchanged, any
# amount of surrounding whitespace), plus a hyphen/en-dash/em-dash *with
# mandatory whitespace on both sides* -- some venues render a multi-author
# byline as "First Last - First Last - First Last" instead of the far more
# common comma-separated form. The whitespace requirement is deliberate: a
# bare "-" with no surrounding space is an ordinary hyphenated single name
# ("Jean-Paul Sartre"), already accepted whole by `_NAME_TOKEN_RE`, and must
# never be split apart.
_NAME_GROUP_SEP_RE = re.compile(r"\s*(?:,|;|\band\b|&)\s*|\s+[-–—]\s+")
_FOOTNOTE_MARKERS = "0123456789*†‡§"
# OCR frequently misreads a superscript footnote asterisk/dagger next to an
# author's name as a stray quotation mark -- stripped before the footnote-
# marker check above, same treatment either character would get.
_NAME_QUOTE_CHARS = "\"'“”‘’"
_CORRESPONDING_AUTHOR_RE = re.compile(r"\bcorresponding\s+authors?\b", re.IGNORECASE)
# A trailing "<Place>, <City>, <digits>, <Country>"-shaped run -- the
# generic shape of a mailing address glued onto an author line by OCR (a
# bare 3-7 digit postal/zip code flanked by capitalized place names),
# regardless of any one country's specific address format.
_ADDRESS_TAIL_RE = re.compile(
    r"[A-Z][\w'\-]*(?:[\s-][A-Z][\w'\-]*)*\s*,\s*[A-Z][\w'\-]*(?:\s[A-Z][\w'\-]*)*\s*,\s*"
    r"\d{3,7}\s*,\s*[A-Z][a-zA-Z]+\s*$"
)


def _looks_like_author_line(s: str) -> bool:
    words = s.split()
    if not (2 <= len(words) <= 12):
        return False
    if s[0].isdigit():
        return False
    name_groups = _NAME_GROUP_SEP_RE.split(s)
    saw_any_name = False
    for name in name_groups:
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        if not (1 <= len(parts) <= 4):
            return False
        for p in parts:
            core = p.rstrip(_FOOTNOTE_MARKERS)
            if not core or not _NAME_TOKEN_RE.match(core):
                return False
        saw_any_name = True
    return saw_any_name


def _name_group_valid(name: str) -> bool:
    """Same per-group name-shape test `_looks_like_author_line` uses
    (1-4 Title-Case tokens), factored out so the compound-line check below
    can walk name groups one at a time instead of requiring the whole
    line to qualify at once."""
    name = name.strip()
    if not name:
        return False
    parts = name.split()
    if not (1 <= len(parts) <= 4):
        return False
    for p in parts:
        core = p.strip(_NAME_QUOTE_CHARS).rstrip(_FOOTNOTE_MARKERS)
        if not core or not _NAME_TOKEN_RE.match(core):
            return False
    return True


def _looks_like_compound_author_line(s: str) -> bool:
    """OCR sometimes glues author names directly onto trailing address/
    corresponding-author text with no detectable line break between fields
    (a layout artifact of a tightly-packed title block, not a vocabulary
    pattern) -- `_looks_like_author_line` correctly requires the *whole*
    line to be name-shaped and so correctly declines this shape. This is
    the narrower, corroborated fallback: at least 2 leading comma/and-
    separated groups must be name-shaped (one capitalized word starting a
    sentence is not evidence by itself), AND the remaining tail -- from the
    first non-name-shaped group onward -- must independently look like a
    corresponding-author marker, an email, or a mailing-address tail. Shape
    only; no author/institution vocabulary of any kind."""
    if not s or s[0].isdigit():
        return False
    groups = [g for g in _NAME_GROUP_SEP_RE.split(s) if g.strip()]
    valid_run = 0
    for g in groups:
        if _name_group_valid(g):
            valid_run += 1
        else:
            break
    if valid_run < 2 or valid_run >= len(groups):
        return False
    remainder = ", ".join(groups[valid_run:])
    return bool(
        _CORRESPONDING_AUTHOR_RE.search(remainder)
        or _EMAIL_RE.search(remainder)
        or _ADDRESS_TAIL_RE.search(remainder)
    )


_REFERENCES_SECTION_RE = re.compile(
    r"^(references|bibliography|works cited|citations|cited works)\s*$", re.IGNORECASE,
)

# Closed, generic vocabulary of academic manuscript-type/section labels
# (spec section 3B): these commonly sit right next to the author/affiliation
# block on page 1 (as a running category tag, e.g. "Review Article",
# "Case Report"), but they name the *submission type*, not a person -- a
# publisher-defined label, not a document-specific string. Checked before
# the author-shape tests below so a 1-4-word label that happens to be
# Title-Case (exactly the shape `_looks_like_author_line` matches on) is
# never misread as a name.
_ARTICLE_TYPE_LABELS = {
    "review article", "research article", "original article", "case report",
    "case study", "short communication", "brief report", "brief communication",
    "letter to the editor", "editorial", "systematic review", "meta-analysis",
    "clinical trial", "technical note", "commentary", "correspondence",
    "erratum", "corrigendum", "original research", "rapid communication",
    "point of view", "perspective", "opinion", "mini review", "invited review",
    "editor's note", "guest editorial", "position paper", "practice guideline",
}


def _is_article_type_label(s: str) -> bool:
    return s.strip().lower().rstrip(".:") in _ARTICLE_TYPE_LABELS


def is_references_heading(text: str) -> bool:
    """True for a References/Bibliography-style section heading. A
    reference-list entry is *expected* to look author-name-shaped,
    conference-shaped, or DOI-shaped -- that is its entire, correct
    content, not front matter that leaked into unrelated body prose -- so
    callers must not apply the "front matter embedded in body" checks
    (validation.py's `_validate_paragraph_continuity`, this module's own
    `repair_interrupted_paragraphs` equivalent) to a section headed like
    this at all."""
    return bool(_REFERENCES_SECTION_RE.match(text.strip()))


_TOC_SECTION_RE = re.compile(
    r"^(table of contents|contents|toc)\s*$", re.IGNORECASE,
)


def is_toc_heading(text: str) -> bool:
    """True for a Table-of-Contents-style section heading. Same shape of
    signal as `is_references_heading` above -- used by retrieval_view.py's
    MAIN_TEXT filter to drop a document's TOC section (entries duplicate
    real headings and add no independent search value) without needing a
    dedicated TableOfContentsBlock type."""
    return bool(_TOC_SECTION_RE.match(text.strip()))


_ABSTRACT_SECTION_RE = re.compile(r"^(abstract|summary)\s*$", re.IGNORECASE)


def is_abstract_heading(text: str) -> bool:
    """True for a standalone "Abstract"/"Summary" section heading (spec
    EXCLUDE "abstract"). Same shape of signal as `is_references_heading` --
    used by retrieval_view.py's MAIN_TEXT filter to drop a document's
    Abstract section (and everything nested under it) the same way it
    already drops References/TOC, without needing every abstract body
    paragraph to have been individually typed as front matter at
    extraction time (`AbstractBlock` is reserved for the much rarer inline
    case -- see model.py -- the common case is a normal H2 `HeadingBlock`
    with ordinary `ParagraphBlock`s underneath it, exactly like any other
    named section)."""
    return bool(_ABSTRACT_SECTION_RE.match(text.strip()))


_ACKNOWLEDGEMENTS_SECTION_RE = re.compile(
    r"^(acknowledge?ments?)\s*$", re.IGNORECASE,
)


def is_acknowledgements_heading(text: str) -> bool:
    """True for an Acknowledgements/Funding section heading (spec EXCLUDE
    "acknowledgements"). Same shape of signal as `is_references_heading`;
    both US ("acknowledgments") and UK ("acknowledgements") spellings are
    generic vocabulary, not tied to any one publisher's wording."""
    return bool(_ACKNOWLEDGEMENTS_SECTION_RE.match(text.strip()))


def classify_front_matter_line(text: str) -> Optional[str]:
    """Returns one of "contact", "permission", "copyright", "conference",
    "publication_metadata", "keywords", "affiliation", "author", or None
    (not front matter).
    Order matters: more specific/stronger signals are checked first so an
    author line that happens to include an email is classified by the
    stronger "contact" signal rather than falling through."""
    s = text.strip()
    if not s:
        return None
    words = s.split()

    if _EMAIL_RE.search(s) and len(words) <= 12:
        return "contact"
    if _PERMISSION_RE.match(s):
        return "permission"
    if _COPYRIGHT_RE.search(s) and len(words) <= 40:
        return "copyright"
    if (_CONFERENCE_RE.search(s) or _CONFERENCE_DATE_TAIL_RE.search(s)) and len(words) <= 40:
        return "conference"
    if _ISBN_DOI_RE.search(s) and len(words) <= 40:
        return "publication_metadata"
    if _KEYWORDS_RE.match(s):
        return "keywords"
    if _AFFIL_KEYWORDS_RE.search(s) and len(words) <= 14 and not s.rstrip().endswith(("!", "?")):
        return "affiliation"
    if _is_article_type_label(s):
        # A manuscript-type/section tag ("Review Article", "Case Report") is
        # publication metadata about the submission, not a name -- classified
        # as such (rather than `None`/not-front-matter) so it's still pulled
        # into the front-matter group instead of being left behind as an
        # orphan paragraph ahead of the title.
        return "publication_metadata"
    if _looks_like_author_line(s):
        return "author"
    if _looks_like_compound_author_line(s):
        return "author"
    return None
