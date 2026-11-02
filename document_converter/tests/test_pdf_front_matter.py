"""
End-to-end PDF-extraction tests for the corrective-implementation-prompt
regression class: title resolution, front-matter grouping, reading order,
the missed-section-heading audit, layout-hyphenation repair, and OCR
bullet-glyph normalization (spec sections 2, 5, 6, 7, 9, 11, 12).

Builds a synthetic multi-page-shaped academic-paper-style PDF with fitz
(no external file, no OCR/tesseract dependency) that reproduces every
failure mode named in the corrective prompt: a wrapped two-line title, a
first author before the abstract, a second author/affiliation and a
permission/copyright notice both interleaved into the introduction, a
missed (body-font) numbered section heading, layout-hyphenated words, true
compound hyphens, and OCR-bullet-style list lines.
"""
from pathlib import Path

import fitz
import pytest

from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.config import OCRConfig
from document_converter.normalize import normalize_document
from document_converter.heading_repair import repair_bodyless_heading_chains
from document_converter.heading_sibling_audit import audit_numbered_sibling_headings
from document_converter.render import render_markdown
from document_converter.model import (
    HeadingBlock, AuthorBlock, AffiliationBlock, ContactBlock, ListBlock,
)

OCR_OFF = OCRConfig(mode="off")


def _build_regression_fixture(path: Path) -> None:
    d = fitz.open()
    p = d.new_page(width=612, height=792)

    p.insert_text((100, 72), "Introducing Computer Systems from a Programmer's", fontsize=18)
    p.insert_text((155, 94), "Perspective", fontsize=18)

    p.insert_text((240, 130), "Randal E. Bryant", fontsize=11)
    p.insert_text((190, 148), "Carnegie Mellon University", fontsize=10)
    p.insert_text((210, 164), "bryant@cs.cmu.edu", fontsize=10)

    p.insert_text((72, 200), "Abstract", fontsize=13)
    p.insert_text((72, 220), "This paper describes an advanced systems course that gives students a", fontsize=10)
    p.insert_text((72, 234), "more complete view of concepts implemented at the machine-level.", fontsize=10)

    p.insert_text((72, 268), "1 Introduction", fontsize=13)
    p.insert_text((72, 290), "Unfortunately, most computer science and computer engineering curricula", fontsize=10)
    p.insert_text((72, 304), "do not have a course that presents this ad- vanced view.", fontsize=10)

    p.insert_text((72, 328), "Permission to make digital or hard copies of part of this work for", fontsize=8)
    p.insert_text((72, 340), "personal use is granted without fee.", fontsize=8)

    p.insert_text((72, 360), "David R. O'Hallaron", fontsize=11)
    p.insert_text((72, 374), "Carnegie Mellon University", fontsize=10)

    p.insert_text((72, 394), "Copyright 2024 ACM. All rights reserved.", fontsize=8)

    p.insert_text((72, 418), "With the ever-increasing use of the Internet and networking, we cover", fontsize=10)
    p.insert_text((72, 432), "these topics net- working stacks over two courses in most curricula.", fontsize=10)

    p.insert_text((72, 460), "e Present a more complete view of systems.", fontsize=10)
    p.insert_text((72, 474), "e Present systems from a programmer's perspective.", fontsize=10)

    p.insert_text((72, 500), "2 Logistical Issues With the ever-increasing use of systems, we cover", fontsize=10)
    p.insert_text((72, 514), "these topics over two courses in most modern curricula today.", fontsize=10)

    p.insert_text((72, 548), "3 Data Representations", fontsize=13)
    p.insert_text((72, 570), "Data is imple- mented using bits, bytes, and words in modern machines.", fontsize=10)

    d.save(str(path))
    d.close()


@pytest.fixture
def regression_doc(tmp_path):
    path = tmp_path / "regression_fixture.pdf"
    _build_regression_fixture(path)
    doc, log = PdfExtractor().extract(path, OCR_OFF)
    log.extend(normalize_document(doc))
    repair_bodyless_heading_chains(doc, log)
    audit_numbered_sibling_headings(doc, log)
    return doc, log


def test_wrapped_title_becomes_single_h1(regression_doc):
    doc, _ = regression_doc
    h1s = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.level == 1]
    assert len(h1s) == 1
    assert h1s[0].text == "Introducing Computer Systems from a Programmer's Perspective"
    assert doc.metadata.title == h1s[0].text


def test_abstract_is_h2_not_h1(regression_doc):
    doc, _ = regression_doc
    abstract_headings = [b for b in doc.blocks if isinstance(b, HeadingBlock) and b.text.strip() == "Abstract"]
    assert len(abstract_headings) == 1
    assert abstract_headings[0].level == 2


def test_title_resolution_evidence_recorded(regression_doc):
    doc, _ = regression_doc
    evidence = doc.metadata.extra.get("title_resolution")
    assert evidence is not None
    assert evidence["source"] == "first_page_visual_title"
    assert evidence["confidence"] > 0.5
    assert evidence["source_block_ids"]


def test_both_authors_grouped_before_abstract_not_in_introduction(regression_doc):
    doc, _ = regression_doc
    kinds = [type(b).__name__ for b in doc.blocks]
    author_positions = [i for i, k in enumerate(kinds) if k == "AuthorBlock"]
    abstract_idx = next(i for i, b in enumerate(doc.blocks)
                         if isinstance(b, HeadingBlock) and b.text.strip() == "Abstract")
    intro_idx = next(i for i, b in enumerate(doc.blocks)
                      if isinstance(b, HeadingBlock) and b.text.strip() == "1 Introduction")
    assert len(author_positions) == 2
    assert all(i < abstract_idx < intro_idx for i in author_positions)
    # Contiguous run: author/affiliation/contact never interleaved with body.
    fm_positions = [i for i, k in enumerate(kinds) if k in
                    ("AuthorBlock", "AffiliationBlock", "ContactBlock")]
    assert fm_positions == list(range(fm_positions[0], fm_positions[0] + len(fm_positions)))


def test_second_author_not_stranded_in_introduction_text(regression_doc):
    doc, _ = regression_doc
    intro_idx = next(i for i, b in enumerate(doc.blocks)
                      if isinstance(b, HeadingBlock) and b.text.strip() == "1 Introduction")
    section2_idx = next(i for i, b in enumerate(doc.blocks)
                         if isinstance(b, HeadingBlock) and b.text.strip().startswith("2 Logistical"))
    for b in doc.blocks[intro_idx + 1:section2_idx]:
        assert not isinstance(b, (AuthorBlock, AffiliationBlock, ContactBlock))
        text = getattr(b, "text", "") or getattr(b, "normalized", "")
        assert "O'Hallaron" not in text


def test_copyright_and_permission_do_not_interrupt_body_prose(regression_doc):
    doc, _ = regression_doc
    markdown = render_markdown(doc)
    assert "Copyright 2024 ACM" not in markdown
    assert "Permission to make digital" not in markdown
    # Retained in metadata for traceability rather than silently dropped.
    excluded = doc.metadata.extra.get("excluded_front_matter") or []
    excluded_kinds = {e["kind"] for e in excluded}
    assert "copyright" in excluded_kinds
    assert "permission" in excluded_kinds


def test_introduction_paragraph_continuity_preserved(regression_doc):
    doc, _ = regression_doc
    markdown = render_markdown(doc)
    # The two intro sentences must appear in the same continuous run of
    # text, not split apart by author/affiliation/copyright content.
    first_idx = markdown.index("Unfortunately, most computer science")
    second_idx = markdown.index("With the ever-increasing use of the Internet")
    between = markdown[first_idx:second_idx]
    assert "Carnegie Mellon University" not in between
    assert "O'Hallaron" not in between
    assert "Copyright" not in between


def test_missed_section_heading_recovered(regression_doc):
    doc, _ = regression_doc
    numbered = [b.text.strip() for b in doc.blocks if isinstance(b, HeadingBlock) and b.text[:1].isdigit()]
    assert "2 Logistical Issues" in numbered
    idx_map = {b.text.strip(): i for i, b in enumerate(doc.blocks) if isinstance(b, HeadingBlock)}
    assert idx_map["1 Introduction"] < idx_map["2 Logistical Issues"] < idx_map["3 Data Representations"]
    section2 = next(b for b in doc.blocks if isinstance(b, HeadingBlock) and b.text.strip() == "2 Logistical Issues")
    section1 = next(b for b in doc.blocks if isinstance(b, HeadingBlock) and b.text.strip() == "1 Introduction")
    assert section2.level == section1.level


def test_layout_hyphenation_repaired_and_compounds_preserved(regression_doc):
    doc, _ = regression_doc
    markdown = render_markdown(doc)
    for repaired_word in ("advanced", "networking", "implemented"):
        assert repaired_word in markdown
    for broken in ("ad- vanced", "net- working", "imple- mented"):
        assert broken not in markdown
    assert "machine-level" in markdown


def test_ocr_bullet_glyphs_become_list_items(regression_doc):
    doc, _ = regression_doc
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert lists
    item_texts = [it.text for it in lists[0].items]
    assert any("Present a more complete view" in t for t in item_texts)
    assert any("programmer's perspective" in t for t in item_texts)
    for t in item_texts:
        assert not t.startswith("e ")


def test_single_legitimate_letter_not_converted_to_bullet(tmp_path):
    """A lone line shaped like a glyph-bullet, with no sibling at the same
    indentation, must be left as ordinary prose (spec: run evidence
    required, no hard-coded 'e'/'@' bullet)."""
    path = tmp_path / "single_line.pdf"
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Notes", fontsize=16)
    p.insert_text((72, 110), "Intro paragraph so the next line is not itself the title.", fontsize=11)
    p.insert_text((72, 140), "e This looks like a bullet but has no sibling nearby at all.", fontsize=11)
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    assert not any(isinstance(b, ListBlock) for b in doc.blocks)
    assert any(f.code == "probable_bullet_glyph_not_normalized" for f in log.findings)
