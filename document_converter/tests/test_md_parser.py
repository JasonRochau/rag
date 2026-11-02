from document_converter.md_parser import parse_markdown
from document_converter.model import HeadingBlock, ParagraphBlock, ListBlock, CodeBlock, TableBlock


def test_parses_headings_and_paragraph():
    text = "# Title\n\nSome paragraph text here.\n\n## Sub\n\nMore text.\n"
    doc, log = parse_markdown(text, source_path="x.md")
    kinds = [type(b).__name__ for b in doc.blocks]
    assert kinds == ["HeadingBlock", "ParagraphBlock", "HeadingBlock", "ParagraphBlock"]
    assert doc.blocks[0].level == 1
    assert doc.blocks[0].text == "Title"


def test_fence_aware_parsing_ignores_hashes_inside_code():
    text = "# Title\n\n```python\n# not a heading\nprint('hi')\n```\n"
    doc, log = parse_markdown(text, source_path="x.md")
    kinds = [type(b).__name__ for b in doc.blocks]
    assert kinds == ["HeadingBlock", "CodeBlock"]
    code = [b for b in doc.blocks if isinstance(b, CodeBlock)][0]
    assert "# not a heading" in code.code
    assert code.language == "python"


def test_list_parsing_with_nesting():
    text = "# T\n\n- item one\n- item two\n  - nested\n"
    doc, log = parse_markdown(text, source_path="x.md")
    lists = [b for b in doc.blocks if isinstance(b, ListBlock)]
    assert len(lists) == 1
    items = lists[0].items
    assert len(items) == 2
    assert items[1].children and items[1].children[0].text == "nested"


def test_table_parsing():
    text = "# T\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
    doc, log = parse_markdown(text, source_path="x.md")
    tables = [b for b in doc.blocks if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].header == ["a", "b"]
    assert tables[0].rows == [["1", "2"]]


def test_block_ids_are_stable_and_sequential():
    text = "# T\n\nPara one.\n\nPara two.\n"
    doc, log = parse_markdown(text, source_path="x.md")
    ids = [b.block_id for b in doc.blocks]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
