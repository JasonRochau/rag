"""
Pipeline-level tests for false-heading demotion (heading_demotion.py) wired
into the shared `_build_document()` stage chain (pipeline.py).

The pass used to be called only from `docling_extractor.py`, making it dead
code on the default legacy PDF path; it now runs for every extractor. These
tests therefore exercise it through `DocumentPipeline.preview()` with a
hand-built `Document` (the same block-model fixture style
test_generalization_repairs.py uses) returned by a stub extractor swapped in
for `select_extractor` -- the legacy/shared path end to end: normalize,
running-header repair, bodyless-heading repair, demotion, the heading
audits, chunking, and validation all see the same mutated document.
"""
from __future__ import annotations

import document_converter.pipeline as pipeline_module
from document_converter.findings import FindingLog, Severity
from document_converter.model import (
    Document, DocumentMetadata, HeadingBlock, ParagraphBlock, UnknownBlock,
)


def _mk(cls, block_id, order, **kw):
    return cls(block_id=block_id, order=order, source_format="pdf", **kw)


def _heading(block_id, order, level, text, page=1):
    return _mk(HeadingBlock, block_id, order, level=level, text=text, normalized=text, page=page)


def _para(block_id, order, text, page=1):
    return _mk(ParagraphBlock, block_id, order, text=text, normalized=text, page=page)


class _StubExtractor:
    """Returns a fixed hand-built Document regardless of the source file --
    the same shape `PdfExtractor.extract()` returns on the default path."""

    def __init__(self, document):
        self._document = document

    def extract(self, path, ocr_config, progress=None):
        return self._document, FindingLog()


def _boilerplate_document():
    """A legacy-path-shaped document: a title, a boilerplate label promoted
    to an H2 three times (the interactive-courseware/checkbox-chrome shape
    the pass exists to catch), each with a real body paragraph so no other
    rule (bodyless-chain, table-header-fragment) applies, and one genuine
    numbered section heading with ordinary prose body."""
    blocks = [
        _heading("b0001", 0, 1, "Quarterly Business Report"),
        _para("b0002", 1, "This report covers ordinary business developments for the period."),
        _heading("b0003", 2, 2, "Participation Activity", page=1),
        _para("b0004", 3, "The first activity prompt asks the reader a simple review question.", page=1),
        _heading("b0005", 4, 2, "Participation Activity", page=2),
        _para("b0006", 5, "The second activity prompt continues with another review question.", page=2),
        _heading("b0007", 6, 2, "Participation Activity", page=3),
        _para("b0008", 7, "The third activity prompt finishes the sequence of review questions.", page=3),
        _heading("b0009", 8, 2, "1. Overview of Business", page=4),
        _para("b0010", 9, "The company describes its principal products and markets here in detail.", page=4),
    ]
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.blocks = blocks
    return doc


def test_pipeline_demotes_verbatim_repeated_boilerplate_headings(pipeline, write_source, monkeypatch):
    src = write_source("boilerplate.pdf", "placeholder -- the stub extractor ignores file contents")
    monkeypatch.setattr(
        pipeline_module, "select_extractor", lambda path: _StubExtractor(_boilerplate_document()),
    )

    result = pipeline.preview(str(src))

    demoted = [b for b in result.full_document.blocks
               if isinstance(b, UnknownBlock) and b.style.get("demoted_from_heading")]
    assert len(demoted) == 3
    assert {b.style.get("demoted_reason") for b in demoted} == {"repeated_boilerplate"}
    # Demotion preserves identity/provenance/searchable text, in place.
    assert [b.block_id for b in demoted] == ["b0003", "b0005", "b0007"]
    assert all(b.text == "Participation Activity" for b in demoted)
    assert [b.page for b in demoted] == [1, 2, 3]
    assert [b.order for b in demoted] == [2, 4, 6]

    # The genuine numbered heading survives as a heading.
    survivors = [b for b in result.full_document.blocks if isinstance(b, HeadingBlock)]
    assert [b.text for b in survivors] == ["Quarterly Business Report", "1. Overview of Business"]

    # Stats land alongside the other repair/audit stats in metadata.extra.
    stats = result.full_document.metadata.extra["heading_demotion"]
    assert stats["headings_demoted"] == 3
    assert stats["demoted_by_reason"] == {"repeated_boilerplate": 3}

    # Demotion findings are FIXED-severity repair records, never errors --
    # they must not trip validation's error gate (findings.py:
    # `log.fixed` -> Severity.FIXED; validation.py counts those under
    # stats["repairs"], separate from warnings/errors).
    findings = result.validation_report.findings.findings
    demotion_findings = [f for f in findings if f.code == "false_heading_demoted"]
    assert len(demotion_findings) == 3
    assert all(f.severity == Severity.FIXED and f.repaired for f in demotion_findings)
    assert result.validation_report.stats["repairs"] >= 3
    assert not result.validation_report.findings.has_errors()


def test_pipeline_leaves_unique_genuine_headings_untouched(pipeline, write_source, monkeypatch):
    """Guard against over-reach on the shared path: short, unnumbered, but
    unique headings with real bodies are legitimate and must survive
    demotion unchanged (spec's warned-against 'isolated short line is
    invalid' over-reach)."""
    blocks = [
        _heading("b0001", 0, 1, "A Short Paper"),
        _para("b0002", 1, "An ordinary opening paragraph of body prose for the paper."),
        _heading("b0003", 2, 2, "Methods", page=1),
        _para("b0004", 3, "The methods used are described in ordinary sentences here.", page=1),
        _heading("b0005", 4, 2, "Results", page=2),
        _para("b0006", 5, "The results are summarized in ordinary sentences as well.", page=2),
    ]
    doc = Document(metadata=DocumentMetadata(source_path="x.pdf", source_format="pdf"))
    doc.blocks = blocks
    src = write_source("genuine.pdf", "placeholder")
    monkeypatch.setattr(pipeline_module, "select_extractor", lambda path: _StubExtractor(doc))

    result = pipeline.preview(str(src))

    headings = [b for b in result.full_document.blocks if isinstance(b, HeadingBlock)]
    assert [b.text for b in headings] == ["A Short Paper", "Methods", "Results"]
    stats = result.full_document.metadata.extra["heading_demotion"]
    assert stats["headings_checked"] == 2
    assert stats["headings_demoted"] == 0
    assert not any(f.code == "false_heading_demoted" for f in result.validation_report.findings.findings)
