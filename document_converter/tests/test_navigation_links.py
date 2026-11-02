"""
Static checks for the index.html <-> document_converter.html cross-navigation
(DELIVERABLES.md's UI-integration task). These parse the two top-level HTML
files as plain text -- there's no JS test runner in this repo and installing
one is out of scope for a navigation-link check -- so what's verified here is
the contract the browser actually sees: real relative hrefs, no iframe, no
hardcoded filesystem paths, no `target="_blank"` popping a new window/tab for
the nav controls themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
CONVERTER_HTML = (REPO_ROOT / "document_converter.html").read_text(encoding="utf-8")


def test_index_html_links_to_document_converter():
    assert re.search(r'<a[^>]+href="document_converter\.html"', INDEX_HTML), \
        "index.html must link to document_converter.html via a real relative href"
    assert "Manage Documents" in INDEX_HTML


def test_document_converter_html_links_back_to_index():
    assert re.search(r'<a[^>]+href="index\.html"', CONVERTER_HTML), \
        "document_converter.html must link back to index.html via a real relative href"


def test_neither_page_uses_a_hardcoded_filesystem_path_for_nav():
    # `openMarkdownWindow`'s comment legitimately mentions "file://" (it
    # documents that the existing Blob-URL markdown viewer works when the
    # page itself is opened as a local file) -- what must never appear is an
    # actual `href="file://...` navigation target.
    for content in (INDEX_HTML, CONVERTER_HTML):
        assert 'href="file://' not in content


def test_neither_page_uses_an_iframe():
    for content in (INDEX_HTML, CONVERTER_HTML):
        assert "<iframe" not in content.lower()


def test_nav_links_do_not_force_a_new_tab():
    """The cross-page links themselves must navigate in place -- unrelated
    to `openMarkdownWindow`'s deliberate `window.open` for viewing converted
    markdown, which is a different, pre-existing feature."""
    nav_link = re.search(r'<a[^>]+href="document_converter\.html"[^>]*>', INDEX_HTML).group(0)
    assert 'target="_blank"' not in nav_link
    back_link = re.search(r'<a[^>]+href="index\.html"[^>]*>', CONVERTER_HTML).group(0)
    assert 'target="_blank"' not in back_link
