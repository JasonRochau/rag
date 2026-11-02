import pytest

from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.findings import Severity


def _doc(text):
    doc, _ = parse_markdown(text, source_path="x.md")
    normalize_document(doc)
    return doc


def test_chunks_carry_provenance_prefix():
    doc = _doc("# T\n\n## Sec\n\nSome content here.\n")
    result = MarkdownChunker().chunk(doc, "x.md", ChunkingConfig(), document_id="d1")
    assert result.chunks
    assert result.chunks[0].text.startswith("[x.md >")
    assert "Sec" in result.chunks[0].header_path[-1] or "T" in result.chunks[0].header_path


def test_packs_multiple_paragraphs_into_one_chunk_under_target():
    text = "# T\n\n" + "\n\n".join(f"Paragraph number {i} with a few words." for i in range(5))
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=1000, target_words=1000)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    assert len(result.chunks) == 1


def test_splits_when_exceeding_target():
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + ("word " * 30) for i in range(10))
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=100, target_words=80)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    assert len(result.chunks) > 1
    for c in result.chunks:
        assert c.word_count <= 100 + 20  # heading/provenance overhead budget


def test_strict_mode_force_splits_oversized_paragraph_with_warning():
    # Since the 2026-09-23 policy change there is no strict-mode
    # error+drop branch for oversized paragraphs: EVERY profile routes them
    # through the emergency sentence-boundary split, with a WARNING.
    text = "# T\n\n" + ". ".join(f"Sentence {i} with several padding words here" for i in range(80)) + "."
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=50, target_words=40, strict=True)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    assert not result.findings.has_errors()
    warnings = [f for f in result.findings.findings if f.code == "forced_fallback_split"]
    assert warnings
    assert all(f.severity == Severity.WARNING for f in warnings)
    assert result.forced_fallback_splits > 0
    assert not result.dropped_block_ids
    # The parent/-part-N provenance convention is preserved.
    parent = warnings[0].block_id
    part_ids = result.split_block_provenance[parent]
    assert len(part_ids) > 1
    assert all(pid.startswith(f"{parent}-part-") for pid in part_ids)


def test_non_strict_mode_force_splits_oversized_paragraph():
    text = "# T\n\n" + ". ".join(f"Sentence {i}" for i in range(80)) + "."
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=50, target_words=40, strict=False, allow_emergency_sentence_split=True)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    assert not result.findings.has_errors()
    assert result.forced_fallback_splits > 0
    assert not result.dropped_block_ids


def test_unsplittable_atomic_block_drops_with_warning_and_document_succeeds():
    # Agreed policy (2026-09-23): a block that genuinely cannot be subdivided
    # within budget is dropped from the index with a WARNING -- the document
    # itself succeeds (no error-level finding, rest of the content indexed).
    items = "\n".join(f"- item {i} with some more words to pad it out" for i in range(40))
    text = f"# T\n\n{items}\n\nA normal trailing paragraph that stays indexed.\n"
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=30, target_words=20, strict=True)
    result = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    findings = [f for f in result.findings.findings if f.code == "oversized_atomic_block"]
    assert findings
    assert all(f.severity == Severity.WARNING for f in findings)
    assert not result.findings.has_errors()
    assert result.oversized_atomic_blocks >= 1
    assert result.dropped_block_ids
    assert any("trailing paragraph" in c.text for c in result.chunks)


def test_single_oversized_sentence_is_force_split_into_token_safe_chunks():
    # A paragraph that is ONE sentence longer than the token budget: sentence
    # -boundary splitting alone cannot help it. It now gets a token-level
    # forced split (`_forced_split_line`) instead of tripping the final
    # `chunk_exceeds_token_limit` guard.
    sentence = " ".join(f"word{i}" for i in range(600))  # no sentence boundary anywhere
    doc = _doc(f"# T\n\n{sentence}\n")
    config = ChunkingConfig(hard_max_words=500, target_words=400, strict=True)
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=512,
    )
    assert not result.findings.has_errors()
    assert result.forced_fallback_splits > 0
    assert not result.dropped_block_ids
    assert len(result.chunks) > 1
    for c in result.chunks:
        assert c.token_count <= 512
    # Every word of the original sentence survives, in order, none duplicated.
    rebuilt = " ".join(c.body.split("\n\n", 1)[1] for c in result.chunks)
    assert rebuilt.split() == sentence.split()


def test_chunk_ids_are_stable_across_runs():
    doc = _doc("# T\n\n## Sec\n\nSome content here.\n")
    config = ChunkingConfig()
    r1 = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    r2 = MarkdownChunker().chunk(doc, "x.md", config, document_id="d1")
    assert [c.chunk_id for c in r1.chunks] == [c.chunk_id for c in r2.chunks]


# ---------------------------------------------------------------------- #
# Token-native target/max packing (embedding-model migration: BGE's
# word-derived ratio path is untouched when target_tokens/max_tokens are
# unset; a Qwen-configured pipeline sets them directly).

def _word_token_counter(text: str) -> int:
    # Deterministic stand-in for a real tokenizer -- 1 token per word, so
    # word-count and token-count assertions can be compared directly.
    return len(text.split())


def test_unset_target_and_max_tokens_preserves_legacy_ratio_behavior():
    """The BGE/legacy path (target_tokens=None, max_tokens=None) must
    behave identically whether or not this migration's code exists --
    packing is still governed by the live service's max_sequence_tokens and
    the word-based target/hard ratio, unchanged."""
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + ("word " * 30) for i in range(10))
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=100, target_words=80)  # target_tokens/max_tokens default None
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=512,
    )
    assert len(result.chunks) > 1
    for c in result.chunks:
        assert c.token_count <= 512


def test_explicit_max_tokens_is_capped_by_live_service_limit():
    """A configured operating max_tokens larger than what the live
    embedding service actually accepts must never produce a chunk the
    service would reject -- the smaller of the two always wins."""
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + ("word " * 30) for i in range(10))
    doc = _doc(text)
    config = ChunkingConfig(hard_max_words=100000, target_words=100000, target_tokens=4096, max_tokens=4096)
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=200,  # live service caps much lower
    )
    for c in result.chunks:
        assert c.token_count <= 200


def test_explicit_target_tokens_governs_packing_size_directly():
    """With target_tokens set (Qwen path) and a generous max_tokens/
    max_sequence_tokens, packing should aim for target_tokens rather than
    the word-ratio-derived target -- verified by packing far more content
    into one chunk than the legacy word-based target would ever allow."""
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + ("word " * 30) for i in range(10))
    doc = _doc(text)
    config = ChunkingConfig(
        hard_max_words=100000, target_words=100000, target_tokens=1000, max_tokens=4096,
    )
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=40960,
    )
    # All 10 paragraphs (~320 tokens total) comfortably fit under a
    # 1000-token target -- must all land in a single chunk.
    assert len(result.chunks) == 1
    assert result.chunks[0].token_count <= 1000


def test_max_tokens_is_a_safety_guard_not_a_required_chunk_size():
    """A short section must not be padded up toward max_tokens -- it's a
    ceiling, never a target chunks are forced to reach."""
    doc = _doc("# T\n\n## Sec\n\nOne short paragraph.\n")
    config = ChunkingConfig(target_tokens=1000, max_tokens=4096)
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=40960,
    )
    assert len(result.chunks) == 1
    assert result.chunks[0].token_count < 100  # nowhere near the 4096 ceiling


def test_never_silently_truncates_retained_text_regardless_of_old_bge_limit():
    """A single long paragraph that would have exceeded the old 512-token
    BGE ceiling must survive whole -- and unsplit -- once packed against a
    Qwen-sized operating budget, instead of being dropped or truncated."""
    long_paragraph = " ".join(f"This is sentence number {i} of a long paragraph." for i in range(120))
    doc = _doc(f"# T\n\n## Sec\n\n{long_paragraph}\n")
    config = ChunkingConfig(
        hard_max_words=100000, target_words=100000, target_tokens=1000, max_tokens=4096, strict=False,
    )
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1",
        token_counter=_word_token_counter, max_sequence_tokens=40960,
    )
    assert not result.findings.has_errors()
    assert not result.dropped_block_ids
    # Every retained word survives across the chunk(s), verbatim.
    reconstructed = " ".join(result.chunks[0].body.split("\n\n", 1)[1].split())
    assert reconstructed == long_paragraph.replace("  ", " ")
