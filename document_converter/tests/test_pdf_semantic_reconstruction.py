"""
PDF-extractor integration tests for the document-agnostic semantic
reconstruction fixes: density-based code-run segmentation with a
prose-sentence run-breaker (Fix 1), literal-markup-example detection
(Fix 4's extractor half), page-region-aware axis/legend heading rejection
(Fix 2), and compound numbered-heading recovery (Fix 3).

All fixtures here are synthetic PDFs built on the fly with PyMuPDF -- no
vocabulary, phrases, or layout specifics are borrowed from any real document;
line content is deliberately generic filler prose plus the failure *shapes*
described in the fix spec (a diagram label between two paragraphs, a literal
HTML example, a compound "N.N Title. Sentence..." line).
"""
from __future__ import annotations

import io
from pathlib import Path

import fitz

from document_converter.extractors.pdf_extractor import PdfExtractor
from document_converter.config import OCRConfig
from document_converter.model import CodeBlock, HeadingBlock, ParagraphBlock

OCR_OFF = OCRConfig(mode="off")


def _build_pdf(tmp_path: Path, name: str, lines: list, font_size: float = 11.0) -> Path:
    """Single-column PDF where every line shares one font size, so nothing
    is heading-shaped by font alone -- exercises the code-run/paragraph
    classification path exactly as an OCR'd (uniformly-sized) page would."""
    d = fitz.open()
    p = d.new_page()
    y = 72
    for line in lines:
        p.insert_text((72, y), line, fontsize=font_size)
        y += 16
    path = tmp_path / name
    d.save(str(path))
    d.close()
    return path


# ---------------------------------------------------------------------- #
# Fix 1: density-based segmentation, prose-sentence run-breaker

def test_prose_figure_labels_prose_does_not_produce_one_giant_code_block(tmp_path):
    lines = [
        "Introduction to the evaluation methodology used throughout this section of the paper.",
        "The system was evaluated using a held-out benchmark set of documents for comparison.",
        "Training",
        "Accuracy",
        "Optimal",
        "This section presents the results of our experiments in significant detail for readers.",
        "Every configuration was run five times and the results were averaged across all trials.",
    ]
    path = _build_pdf(tmp_path, "prose_labels_prose.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]

    # I1 (held-out invariant): no atomic code block may contain multiple
    # complete prose sentences -- the old "one strong signal anywhere"
    # rule would have swept the whole run (including both paragraphs) into
    # a single CodeBlock just because "Training"/"Accuracy"/"Optimal" sat
    # between them.
    for cb in code_blocks:
        sentence_count = cb.code.count(". ") + (1 if cb.code.rstrip().endswith(('.', '!', '?')) else 0)
        assert sentence_count < 3, f"code block absorbed too much prose: {cb.code!r}"

    joined_paragraphs = " ".join(p.text for p in paragraphs)
    assert "evaluation methodology" in joined_paragraphs
    assert "averaged across all trials" in joined_paragraphs


def test_real_algorithm_pseudocode_still_grouped_as_code(tmp_path):
    lines = [
        "The following pseudocode summarizes the traversal procedure used by the algorithm.",
        "1: function TRAVERSE(node)",
        "2: visited[node] — true",
        "3: for all children c of node do",
        "4: if not visited[c] then",
        "5: TRAVERSE(c)",
        "6: end if",
        "7: end for",
        "8: end function",
        "The procedure above terminates once every reachable node has been visited exactly once.",
    ]
    path = _build_pdf(tmp_path, "algorithm.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code_blocks, "numbered algorithm lines should still be grouped into an atomic code block"
    combined_code = "\n".join(cb.code for cb in code_blocks)
    assert "TRAVERSE" in combined_code
    assert "end function" in combined_code
    # The surrounding prose sentences must not have been absorbed.
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert any("pseudocode summarizes" in p.text for p in paragraphs)
    assert any("terminates once" in p.text for p in paragraphs)


def test_code_with_spaced_calls_and_bare_assignments_still_detected(tmp_path):
    """Regression for a real density-threshold gap found while validating
    Fix 1 against a held-out fixture: a function-call written as "name (args)"
    (space before the paren -- common in print-to-PDF/OCR'd C-like code) and
    bare assignment lines ("i=0", "total = 8") carried no signal at all in
    the original narrow scanner, so a whole real code block silently fell
    back to plain paragraphs once density-based segmentation replaced the
    old "any one signal marks everything" rule."""
    lines = [
        "The function below searches a list from the beginning until the key is found.",
        "LinearSearch (numbers, numbersSize, key) {",
        "i=0",
        "for (i = 0; i < numbersSize; ++i) {",
        "if (numbers[i] == key) {",
        "return i",
        "}",
        "return -1 // not found",
        "}",
        "The function returns the index of the key, or -1 if it is not present in the list.",
    ]
    path = _build_pdf(tmp_path, "spaced_call_code.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code_blocks, "code with spaced function calls and bare assignments must still become a code block"
    combined_code = "\n".join(cb.code for cb in code_blocks)
    assert "LinearSearch" in combined_code
    assert "i=0" in combined_code
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert any("searches a list" in p.text for p in paragraphs)
    assert any("returns the index" in p.text for p in paragraphs)


# ---------------------------------------------------------------------- #
# Fix 4 (extractor half): literal markup examples become one html code block

def test_literal_html_example_becomes_atomic_html_block(tmp_path):
    lines = [
        "The following listing shows a minimal document used to illustrate the structure.",
        "<html>",
        "<head><title>Example</title></head>",
        "<body>",
        "<p>Hello world</p>",
        "</body>",
        "</html>",
        "This structure is parsed into a tree by the algorithm described in the next section.",
    ]
    path = _build_pdf(tmp_path, "html_example.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    html_blocks = [b for b in code_blocks if b.language == "html"]
    assert html_blocks, "a run of consecutive markup-tag lines should become one html code block"
    assert "<html>" in html_blocks[0].code and "</html>" in html_blocks[0].code

    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert any("minimal document" in p.text for p in paragraphs)
    assert any("parsed into a tree" in p.text for p in paragraphs)


# ---------------------------------------------------------------------- #
# Fix 3: compound numbered heading recovery

def test_compound_numbered_heading_recovered_as_heading_and_paragraph(tmp_path):
    lines = [
        # A plausible outline path leading up to the compound line (1 ->
        # 3.2.3 with nothing in between would itself be an implausible
        # jump, correctly rejected by outline-sequence validation).
        "3.2.3 Suffix Extraction. For efficiency, the first step is to identify boundaries reliably.",
        "Subsequent steps refine the boundaries using a secondary pass over the input structure.",
    ]
    path = _build_pdf(tmp_path, "compound_heading.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = [h.text.strip() for h in headings]
    assert any(t.startswith("3.2.3") and "Suffix Extraction" in t for t in heading_texts), heading_texts

    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert any(p.text.startswith("For efficiency") for p in paragraphs), [p.text for p in paragraphs]


def test_compound_split_does_not_fire_on_citation_style_numbers(tmp_path):
    lines = [
        "1 Introduction",
        "This paper studies a general problem relevant to a broad class of applications today.",
        "3.1 See Smith et al. for details on this topic and its implications for the broader field.",
        "We build on that foundation throughout the remainder of this paper in several ways.",
    ]
    path = _build_pdf(tmp_path, "compound_citation.pdf", lines)
    doc, log = PdfExtractor().extract(path, OCR_OFF)

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = [h.text.strip() for h in headings]
    assert not any("see smith" in t.lower() for t in heading_texts), heading_texts


# ---------------------------------------------------------------------- #
# Fix 2: axis/legend label rejected only when near figure content

def _tiny_png_bytes() -> bytes:
    from PIL import Image
    img = Image.new("RGB", (40, 40), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_axis_label_near_image_is_not_promoted_to_heading(tmp_path):
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Results", fontsize=16)
    p.insert_text((72, 110), "The chart below summarizes accuracy across all evaluated configurations.", fontsize=11)
    p.insert_image(fitz.Rect(72, 140, 300, 300), stream=_tiny_png_bytes())
    p.insert_text((320, 150), "Accuracy", fontsize=11)
    p.insert_text((72, 320), "Figure 2: Accuracy across configurations", fontsize=9)
    path = tmp_path / "axis_near_image.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = [h.text.strip() for h in headings]
    assert "Accuracy" not in heading_texts, heading_texts


def test_axis_label_word_elsewhere_can_still_be_a_heading(tmp_path):
    """The same bare word, standing alone with no figure/caption nearby,
    must remain a valid heading -- the page-region condition is mandatory,
    not a standing vocabulary rejection (spec Fix 2)."""
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Report", fontsize=16)
    p.insert_text((72, 100), "This report presents an overview of the proposed method and its evaluation.", fontsize=11)
    p.insert_text((72, 120), "The remainder of this document is organized as described in the next section.", fontsize=11)
    p.insert_text((72, 400), "Accuracy", fontsize=14)
    p.insert_text((72, 430), "This section discusses accuracy considerations for the proposed method in depth.", fontsize=11)
    p.insert_text((72, 450), "No figures or tables accompany this particular discussion in the document.", fontsize=11)
    path = tmp_path / "axis_alone.pdf"
    d.save(str(path))
    d.close()

    doc, log = PdfExtractor().extract(path, OCR_OFF)
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    heading_texts = [h.text.strip() for h in headings]
    assert "Accuracy" in heading_texts, heading_texts
