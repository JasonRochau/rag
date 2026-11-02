from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document
from document_converter.model import HeadingBlock


def _headings(doc):
    return [b for b in doc.blocks if isinstance(b, HeadingBlock)]


def test_synthesizes_missing_h1():
    doc, _ = parse_markdown("## Only a subheading\n\ntext\n", source_path="my_doc.md")
    log = normalize_document(doc)
    h = _headings(doc)
    assert h[0].level == 1
    assert any(f.code == "synthesized_h1" for f in log.findings)


def test_demotes_extra_h1s():
    doc, _ = parse_markdown("# First\n\ntext\n\n# Second\n\nmore\n", source_path="x.md")
    log = normalize_document(doc)
    h = _headings(doc)
    assert [x.level for x in h] == [1, 2]
    assert any(f.code == "demoted_extra_h1" for f in log.findings)


def test_repairs_skipped_heading_level():
    doc, _ = parse_markdown("# T\n\n#### Too deep\n\ntext\n", source_path="x.md")
    log = normalize_document(doc)
    h = _headings(doc)
    assert [x.level for x in h] == [1, 2]
    assert any(f.code == "skipped_heading_level" for f in log.findings)


def test_noop_when_already_well_formed():
    doc, _ = parse_markdown("# T\n\ntext\n\n## Sub\n\nmore\n", source_path="x.md")
    log = normalize_document(doc)
    assert log.findings == []
