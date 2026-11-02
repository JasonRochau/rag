"""Unit tests for text_repair.py (spec sections 11/12): layout-hyphenation
repair and OCR bullet-glyph line-shape matching. Pure functions, no PDF
involved."""
from document_converter.text_repair import (
    HyphenationStats, repair_layout_hyphenation, ocr_bullet_glyph_match,
)


def test_layout_hyphenation_repaired_for_true_wraps():
    stats = HyphenationStats()
    cases = {
        "This is an ad- vanced concept.": "This is an advanced concept.",
        "We imple- mented it.": "We implemented it.",
        "This uses net- working stacks.": "This uses networking stacks.",
        "These con- cepts matter.": "These concepts matter.",
    }
    for original, expected in cases.items():
        assert repair_layout_hyphenation(original, stats) == expected
    assert stats.layout_hyphenation_repairs == 4
    assert stats.unresolved_layout_hyphenation == 0


def test_true_compound_hyphens_preserved():
    stats = HyphenationStats()
    cases = [
        "A machine- level abstraction.",
        "A client- server model.",
        "A programmer- centric view.",
        "An upper- level API.",
        "A bit- level trick.",
    ]
    for text in cases:
        repaired = repair_layout_hyphenation(text, stats)
        # Spacing is normalized but the hyphen itself must survive.
        assert "-" in repaired
        assert " - " not in repaired and "- " not in repaired
    assert stats.layout_hyphenation_rejected == len(cases)
    assert stats.layout_hyphenation_repairs == 0


def test_ordinary_inline_hyphen_untouched():
    stats = HyphenationStats()
    text = "This uses a client-server model with no line wrap at all."
    assert repair_layout_hyphenation(text, stats) == text
    # No whitespace after the hyphen -- never even a candidate.
    assert stats.layout_hyphenation_candidates == 0


def test_hyphenation_repair_log_has_source_block_ids():
    stats = HyphenationStats()
    repair_layout_hyphenation("We imple- mented it.", stats, block_id="b0042")
    assert stats.repairs_log[0]["block_id"] == "b0042"
    assert stats.repairs_log[0]["action"] == "repaired"


def test_ocr_bullet_glyph_match_shapes():
    assert ocr_bullet_glyph_match("e Present a more complete view of systems.") == \
        "Present a more complete view of systems."
    assert ocr_bullet_glyph_match("@ It exposes students to techniques and tools.") == \
        "It exposes students to techniques and tools."


def test_ocr_bullet_glyph_match_rejects_ordinary_prose():
    assert ocr_bullet_glyph_match("Experience with systems matters a great deal.") is None
    assert ocr_bullet_glyph_match("e.g. this is not a bullet at all") is None
