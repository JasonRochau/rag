from pathlib import Path

from document_converter.config import OCRConfig
from document_converter.extractors import select_extractor
from document_converter.extractors.txt_extractor import TxtExtractor
from document_converter.extractors.html_extractor import HtmlExtractor
from document_converter.extractors.markdown_extractor import MarkdownExtractor
from document_converter.model import HeadingBlock, ParagraphBlock, ListBlock, CodeBlock, TableBlock, ImageBlock

OCR_OFF = OCRConfig(mode="off")


def test_select_extractor_routes_by_extension(tmp_path):
    for name, expected in [
        ("a.md", MarkdownExtractor), ("a.markdown", MarkdownExtractor),
        ("a.txt", TxtExtractor), ("a.html", HtmlExtractor), ("a.htm", HtmlExtractor),
    ]:
        p = tmp_path / name
        p.write_text("hello")
        assert isinstance(select_extractor(p), expected)


def test_txt_extractor_detects_title_and_paragraphs(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("My Title\n\nThis is a paragraph of plain text content.\n")
    doc, log = TxtExtractor().extract(p, OCR_OFF)
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert headings and headings[0].text == "My Title"
    assert paragraphs


def test_txt_extractor_detects_bullet_list(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Title\n\n- item one\n- item two\n")
    doc, log = TxtExtractor().extract(p, OCR_OFF)
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert lists and len(lists[0].items) == 2


def test_html_extractor_strips_boilerplate_and_keeps_content(tmp_path):
    html = """
    <html><head><title>Doc Title</title></head>
    <body>
      <nav>skip me</nav>
      <h1>Heading One</h1>
      <p>A paragraph with <a href="http://x.test">a link</a>.</p>
      <ul><li>one</li><li>two</li></ul>
      <pre><code class="language-python">print(1)</code></pre>
      <script>evil()</script>
    </body></html>
    """
    p = tmp_path / "page.html"
    p.write_text(html)
    doc, log = HtmlExtractor().extract(p, OCR_OFF)
    assert doc.metadata.title == "Doc Title"
    kinds = [type(b).__name__ for b in doc.blocks]
    assert "HeadingBlock" in kinds
    assert "ListBlock" in kinds
    code_blocks = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code_blocks and code_blocks[0].language == "python"
    paragraphs = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert any("[a link]" in p.text for p in paragraphs)
    # boilerplate must not leak through as content
    assert not any("skip me" in (getattr(b, "text", "") or "") for b in doc.blocks)
    assert not any("evil()" in (getattr(b, "text", "") or getattr(b, "code", "")) for b in doc.blocks)


def test_html_extractor_flags_complex_table(tmp_path):
    html = "<table><tr><th>a</th><th>b</th></tr><tr><td colspan=2>x</td></tr></table>"
    p = tmp_path / "t.html"
    p.write_text(f"<html><body>{html}</body></html>")
    doc, log = HtmlExtractor().extract(p, OCR_OFF)
    tables = [b for b in doc.blocks if isinstance(b, TableBlock)]
    assert tables and tables[0].is_complex
    assert any(f.code == "complex_html_table" for f in log.findings)


def test_markdown_extractor_roundtrips_through_common_parser(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome text.\n")
    doc, log = MarkdownExtractor().extract(p, OCR_OFF)
    assert doc.metadata.source_format == "markdown"
    assert isinstance(doc.blocks[0], HeadingBlock)
