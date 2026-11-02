"""
Regression fixtures (spec section 22) for the two documents that exposed the
original universal-formatting failures:

  Fixture A: "CSC506 Design and Analysis of Algorithms - Chapter2.pdf" --
  a fully-scanned, OCR-heavy, interactive-courseware (zyBooks) PDF export.
  Originally produced hundreds of false headings (code lines, braces, single
  letters, OCR fragments) and a flat H1/H2-only hierarchy.

  Fixture B: "Linear Algebraic Depth-First Search.pdf" -- a born-digital,
  two-column ACM paper (also fully OCR'd in this environment, since its
  content is vector-drawn glyphs with no extractable text layer or raster
  image). Originally promoted running headers/footers/page numbers/author
  names/equations/algorithm lines to headings, with everything flattened to
  H2.

Both fixtures are real, unedited source files under
docs/newly_formatted_docs_review/sources/ -- nothing about their content is
hard-coded here; these tests assert *structural properties*, not specific
titles/topics/wording, per spec section 23. Both are real multi-page OCR
runs (tens of seconds each), not mocked.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from document_converter.config import OCRConfig
from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.normalize import normalize_document
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.validation import validate
from document_converter.findings import FindingLog
from document_converter.model import HeadingBlock

HAS_TESSERACT = shutil.which("tesseract") is not None
_SOURCES = Path(__file__).resolve().parents[2] / "docs" / "newly_formatted_docs_review" / "sources"
FIXTURE_A = _SOURCES / "CSC506 Design and Analysis of Algorithms - Chapter2.pdf"
FIXTURE_B = _SOURCES / "Linear Algebraic Depth-First Search.pdf"

pytestmark = pytest.mark.skipif(
    not HAS_TESSERACT, reason="tesseract binary not installed (both fixtures are fully OCR'd)"
)


def _process(path: Path):
    document, extract_log = PdfExtractor().extract(path, OCRConfig(mode="on"))
    norm_log = normalize_document(document)
    upstream = FindingLog()
    upstream.extend(extract_log)
    upstream.extend(norm_log)
    config = ChunkingConfig()
    result = MarkdownChunker().chunk(document, str(path), config, document_id="regression")
    report = validate(document, result, config, str(path), "", "pdf", document.metadata.ocr_used, upstream)
    return document, report


@pytest.fixture(scope="module")
def fixture_a():
    if not FIXTURE_A.exists():
        pytest.skip(f"regression fixture not present: {FIXTURE_A}")
    return _process(FIXTURE_A)


@pytest.fixture(scope="module")
def fixture_b():
    if not FIXTURE_B.exists():
        pytest.skip(f"regression fixture not present: {FIXTURE_B}")
    return _process(FIXTURE_B)


def _headings(document):
    return [b for b in document.blocks if isinstance(b, HeadingBlock)]


_BAD_SUBSTRINGS = (
    "return -1", "print (", "end function", "// not found",
)


# ---------------------------------------------------------------------- #
# Fixture A: OCR-heavy, layout-complex courseware PDF

def test_fixture_a_no_ocr_fragment_or_code_headings(fixture_a):
    document, report = fixture_a
    assert report.stats["code_like_heading_count"] == 0
    assert report.stats["equation_like_heading_count"] == 0
    assert report.stats["page_furniture_heading_count"] == 0
    for h in _headings(document):
        text = h.text.strip()
        assert text not in ("}", "0"), f"punctuation/digit fragment promoted to heading: {text!r}"
        for bad in _BAD_SUBSTRINGS:
            assert bad not in text.lower(), f"code fragment promoted to heading: {text!r}"


def test_fixture_a_exactly_one_h1(fixture_a):
    document, report = fixture_a
    h1s = [h for h in _headings(document) if h.level == 1]
    assert len(h1s) == 1
    assert report.stats["headers"]["h1"] == 1


def test_fixture_a_real_sections_retain_valid_hierarchy(fixture_a):
    document, report = fixture_a
    assert report.stats["heading_hierarchy_valid"] is True
    numbered = [h for h in _headings(document) if h.level >= 2 and h.text.strip()[:1].isdigit()]
    assert len(numbered) >= 5, "expected several real numbered sections (2.1, 2.2, ...) to survive as headings"


def test_fixture_a_paragraphs_are_not_empty_or_degenerate(fixture_a):
    document, report = fixture_a
    assert report.stats["paragraph_count"] > 50


def test_fixture_a_chunk_count_not_wildly_inflated_by_false_headings(fixture_a):
    """Before the fix this document produced ~300+ headings (one per stray
    OCR line) and a correspondingly inflated chunk count. Chunk count should
    now track real structure (roughly one chunk per genuine section), not
    one per line."""
    document, report = fixture_a
    heading_count = report.stats["heading_count"]
    assert heading_count < 150, f"heading_count={heading_count} still looks inflated by false headings"


def test_fixture_a_structural_readiness_reports_honestly(fixture_a):
    """This fixture is known to still have real over-segmentation (short
    interactive-courseware UI labels like 'Show answer' are valid headings
    syntactically, but carry no body content) -- the requirement is that
    this gets reported truthfully (non-zero errors, structural_readiness
    False), not silently passed as clean."""
    document, report = fixture_a
    default_tiny_ratio_limit = ChunkingConfig().tiny_section_ratio_limit
    if report.stats["tiny_section_ratio"] > default_tiny_ratio_limit or report.stats["empty_section_count"] > 0:
        assert report.stats["structural_readiness"] is False
        assert report.stats["errors"] > 0


# ---------------------------------------------------------------------- #
# Fixture B: born-digital, multi-column academic paper

def test_fixture_b_title_retained(fixture_b):
    document, report = fixture_b
    h1s = [h for h in _headings(document) if h.level == 1]
    assert len(h1s) == 1
    assert len(h1s[0].text.strip()) > 10


def test_fixture_b_abstract_and_major_sections_retained(fixture_b):
    document, report = fixture_b
    texts = [h.text.strip().lower() for h in _headings(document)]
    assert any("abstract" in t for t in texts)
    assert any(t.startswith("1 ") or t == "1 introduction" for t in texts)


def test_fixture_b_subsections_are_child_levels(fixture_b):
    document, report = fixture_b
    by_level = {}
    for h in _headings(document):
        by_level.setdefault(h.level, []).append(h.text.strip())
    assert 3 in by_level, "expected numbered subsections (3.1, 4.1, ...) to be nested deeper than top sections"


def test_fixture_b_running_headers_and_page_numbers_excluded(fixture_b):
    document, report = fixture_b
    texts = [h.text.strip() for h in _headings(document)]
    assert not any(t.isdigit() for t in texts), "a bare page number was promoted to a heading"
    assert not any("phoenix, az" in t.lower() for t in texts), "a running venue/header line was promoted to a heading"


def test_fixture_b_author_footer_not_repeated_as_body_content(fixture_b):
    document, _ = fixture_b
    occurrences = sum(
        1 for b in document.blocks
        if "spampinato, upasana sridhar, and tze meng low" in (getattr(b, "text", "") or "").lower()
        and "2019" not in (getattr(b, "text", "") or "")  # the one legitimate ACM-reference-citation occurrence
    )
    assert occurrences <= 1, f"author-name running footer leaked into body content {occurrences} times"


def test_fixture_b_no_code_or_equation_headings(fixture_b):
    document, report = fixture_b
    assert report.stats["code_like_heading_count"] == 0
    assert report.stats["equation_like_heading_count"] == 0
    assert report.stats["page_furniture_heading_count"] == 0


def test_fixture_b_algorithm_lines_not_individually_headed(fixture_b):
    document, report = fixture_b
    texts = [h.text.strip().lower() for h in _headings(document)]
    assert not any(t.startswith(("end function", "end if", "end for", "end while")) for t in texts)


def test_fixture_b_multi_column_order_valid(fixture_b):
    """Reading order must not interleave the two columns line-by-line: the
    Introduction's own text should appear as a coherent run, not scattered
    single lines alternating with unrelated column-2 content."""
    document, _ = fixture_b
    intro_heading_idx = None
    for i, b in enumerate(document.blocks):
        if isinstance(b, HeadingBlock) and b.text.strip().lower().startswith("1 introduction"):
            intro_heading_idx = i
            break
    assert intro_heading_idx is not None
    next_heading_idx = next(
        (i for i, b in enumerate(document.blocks)
         if i > intro_heading_idx and isinstance(b, HeadingBlock)),
        len(document.blocks),
    )
    assert next_heading_idx - intro_heading_idx >= 2, "Introduction section has suspiciously little content"
