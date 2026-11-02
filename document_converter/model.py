"""
Common internal document representation.

Every format-specific extractor produces a `Document` made of typed `Block`
subclasses. Nothing downstream (normalization, semantic structuring,
rendering, chunking) talks to a source format directly again -- everything
after extraction operates on this common representation.

Design notes / assumptions (see deliverables doc for the full list):
  * `block_id` is a stable per-document identifier ("b0001", "b0002", ...)
    assigned in source order by the extractor. Semantic structuring refers
    to blocks only by this ID, never by position, so reordering the list
    does not change identity.
  * `word_count` is `len(normalized.split())`; used for chunk packing and
    paragraph-length warnings, everywhere in the pipeline.
  * `source_pages` on a chunk is derived by unioning the `page` field of the
    blocks that made up the chunk -- pages are 1-indexed, absent for
    formats without page semantics (markdown/txt/html).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DocumentMetadata:
    source_path: str
    source_format: str  # "pdf" | "docx" | "html" | "txt" | "markdown"
    title: Optional[str] = None
    page_count: Optional[int] = None
    ocr_used: bool = False
    extraction_warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Block:
    """Base class for every node in the common document representation."""

    block_id: str
    order: int
    source_format: str
    raw: str = ""
    normalized: str = ""
    page: Optional[int] = None
    style: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        # Falls back to this block's own `.text` (when the subclass has one)
        # only if neither `normalized` nor `raw` was populated -- an
        # extractor/adapter that stores real content in `.text` but forgets
        # to mirror it into `.normalized` must never make that content
        # invisible to size-based chunk packing, regardless of which parser
        # produced the block or whether that content happens to contain
        # newlines.
        text = self.normalized or self.raw or getattr(self, "text", "")
        return len(text.split()) if text else 0

    @property
    def kind(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"] = self.kind
        d["word_count"] = self.word_count
        return d


@dataclass
class HeadingBlock(Block):
    level: int = 1
    text: str = ""


@dataclass
class ParagraphBlock(Block):
    text: str = ""


@dataclass
class ListItemNode:
    text: str
    checked: Optional[bool] = None  # task list state, None = not a task item
    children: List["ListItemNode"] = field(default_factory=list)


@dataclass
class ListBlock(Block):
    ordered: bool = False
    items: List[ListItemNode] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        def _count(items: List[ListItemNode]) -> int:
            n = 0
            for it in items:
                n += len(it.text.split())
                n += _count(it.children)
            return n
        return _count(self.items)


@dataclass
class TableBlock(Block):
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    is_complex: bool = False  # flagged for manual review (merged cells etc.)

    @property
    def word_count(self) -> int:
        n = len(" ".join(self.header).split())
        for row in self.rows:
            n += len(" ".join(row).split())
        return n


@dataclass
class CodeBlock(Block):
    language: str = ""
    code: str = ""

    @property
    def word_count(self) -> int:
        # Code is measured in whitespace-split tokens too, for budget purposes.
        return len(self.code.split())


@dataclass
class QuoteBlock(Block):
    text: str = ""


@dataclass
class ImageBlock(Block):
    alt: str = ""
    src: str = ""


@dataclass
class PageBreakBlock(Block):
    pass


@dataclass
class WarningBlock(Block):
    """A block the extractor could not confidently classify but must not drop."""
    text: str = ""


@dataclass
class UnknownBlock(Block):
    text: str = ""


@dataclass
class CaptionBlock(Block):
    """A figure/table caption line ("Figure 3: ...", "Table 1. ..."). Never a
    heading, but real content -- kept searchable and attached to the section
    it appeared in, not merged into surrounding prose or dropped."""
    text: str = ""


@dataclass
class FigureLabelBlock(Block):
    """A label naming a figure/image ("Original Image", "Input Image",
    "Image 2") -- structurally isolated like a heading candidate, but never
    promoted to one."""
    text: str = ""


@dataclass
class DiagramAnnotationBlock(Block):
    """An axis/legend/color/direction annotation on a chart or diagram
    ("Y - Axis", "Color: Red", "North Direction Edges")."""
    text: str = ""


@dataclass
class TableCellBlock(Block):
    """A standalone value/label recovered from what looks like a table row
    or key-value pair in OCR'd text, when it could not be reconstructed into
    a real TableBlock."""
    text: str = ""


@dataclass
class TitleBlock(Block):
    """The resolved document title (spec section 2) -- may combine several
    physically adjacent source lines (a wrapped multi-line title) into one
    logical block. Rendered as the document's single H1; never a candidate
    for any other heading level."""
    text: str = ""


@dataclass
class AuthorBlock(Block):
    """A single author name (or name list) recovered from first-page front
    matter. Grouped with other front-matter blocks after the title, never
    left interleaved with body content (spec section 5)."""
    text: str = ""


@dataclass
class AffiliationBlock(Block):
    """An institutional affiliation line associated with one or more
    authors (university/department/lab name)."""
    text: str = ""


@dataclass
class ContactBlock(Block):
    """An author contact line (email address, and/or other direct contact
    info) recovered from front matter."""
    text: str = ""


@dataclass
class AbstractBlock(Block):
    """Reserved for a non-heading abstract label/marker recovered from
    front matter when it cannot be promoted to a real `Abstract` heading
    (e.g. inline with author text). The common case -- a standalone
    "Abstract" line -- is promoted to a normal H2 `HeadingBlock` instead,
    consistent with the rest of the section-heading pipeline; this type
    exists for the classifier taxonomy (spec section 4) rather than as the
    primary path."""
    text: str = ""


@dataclass
class KeywordBlock(Block):
    """A "Keywords: ..." front-matter line."""
    text: str = ""


@dataclass
class PublicationMetadataBlock(Block):
    """Generic publication metadata (ISBN/DOI, journal/volume/issue,
    publisher name) that is neither a copyright notice, a permission
    notice, nor conference-specific metadata."""
    text: str = ""


@dataclass
class CopyrightNoticeBlock(Block):
    """A copyright line ("(c) 2024 ...", "All rights reserved.")."""
    text: str = ""


@dataclass
class PermissionNoticeBlock(Block):
    """Boilerplate reproduction/permission-to-copy notice text, distinct
    from a copyright line (spec section 4/8)."""
    text: str = ""


@dataclass
class ConferenceMetadataBlock(Block):
    """Conference/venue/proceedings metadata ("Proceedings of the 30th
    ...", a venue-year token like "SOSP '21")."""
    text: str = ""


# Front-matter block types the classifier can produce (spec section 4),
# excluding TitleBlock/AbstractBlock which have dedicated handling above.
FRONT_MATTER_BLOCK_TYPES = (
    AuthorBlock, AffiliationBlock, ContactBlock, KeywordBlock,
    PublicationMetadataBlock, CopyrightNoticeBlock, PermissionNoticeBlock,
    ConferenceMetadataBlock,
)


@dataclass
class Document:
    metadata: DocumentMetadata
    blocks: List[Block] = field(default_factory=list)

    def next_block_id(self) -> str:
        # Derived from the highest numeric suffix actually in use, not
        # `len(self.blocks)`: a repair pass that *removes* blocks (e.g.
        # `running_header_repair.py` dropping a duplicate running header)
        # shrinks the list without renumbering the block_ids that remain,
        # so `len`-based generation can mint an id that collides with one
        # still in use further down the list.
        max_n = 0
        for b in self.blocks:
            m = re.match(r"^b(\d+)$", b.block_id or "")
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"b{max_n + 1:04d}"

    def add(self, block: Block) -> Block:
        self.blocks.append(block)
        return block

    def block_by_id(self, block_id: str) -> Optional[Block]:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        return None
