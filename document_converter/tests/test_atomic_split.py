import os
import random
import re

import pytest

from document_converter.model import CodeBlock, ListBlock, ListItemNode, TableBlock
from document_converter.render import render_block
from document_converter.atomic_split import (
    split_oversized_atomic_block, verify_reconstruction, AtomicSplitError,
)
from document_converter.chunker import MarkdownChunker
from document_converter.config import ChunkingConfig
from document_converter.md_parser import parse_markdown
from document_converter.normalize import normalize_document

_WORD_TOK = lambda s: len(s.split())  # noqa: E731 -- deterministic stand-in tokenizer for these unit tests

# Real subword tokenizer -- the same tokenizer family/model id the live
# embedding service tokenizes with (config.py's EMBEDDING_MODEL default,
# BAAI/bge-large-en-v1.5; embedding_server.py's count_tokens() calls this
# exact tokenizer's .encode()), loaded offline from the local HF cache so
# these tests never hit the network. Table-splitting correctness tests use
# this, never a word count, per the spec's "real tokenizer only" testing
# discipline -- word counts can't surface a tokenizer-merge/BPE artifact
# even in principle (see TABLE_SPLIT_TOKEN_ACCOUNTING_ANALYSIS.md section 2.5).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
try:
    from transformers import AutoTokenizer as _AutoTokenizer
    _real_hf_tokenizer = _AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    def _REAL_TOK(s: str) -> int:
        return len(_real_hf_tokenizer.encode(s, add_special_tokens=True))
except Exception:
    _REAL_TOK = None

requires_real_tokenizer = pytest.mark.skipif(
    _REAL_TOK is None, reason="real subword tokenizer (BAAI/bge-large-en-v1.5) not available offline"
)


def _table_block(header, rows, block_id="tbl"):
    return TableBlock(block_id=block_id, order=0, source_format="pdf", header=header, rows=rows)


def _code_block(lines, block_id="b0271"):
    return CodeBlock(block_id=block_id, order=0, source_format="pdf", page=11, language="", code="\n".join(lines))


def test_oversized_code_block_splits_at_complete_lines():
    lines = [f"stmt_{i} = compute(x_{i}, y_{i}, z_{i})" for i in range(30)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10)
    assert len(parts) > 1
    # Every part's own lines are exact, verbatim original lines -- never a mid-line fragment.
    for p in parts:
        for line in p.block.code.split("\n"):
            assert line in lines


def test_pseudocode_splits_at_algorithm_step_boundaries():
    lines = (
        ["Algorithm 1 does something important with several descriptive words right here"]
        + [f"{i}: step number {i} performs some unit of work in this line" for i in range(1, 20)]
    )
    block = _code_block(lines, block_id="b0273")
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=70, provenance_token_reserve=15)
    assert len(parts) > 1
    for p in parts[1:]:
        first_line = lines[p.line_start - 1]
        assert re.match(r"^\d+:", first_line) or first_line.startswith("Algorithm")


def test_no_part_exceeds_the_real_tokenizer_limit():
    lines = [f"line {i} with a handful of filler words describing step {i}" for i in range(25)]
    block = _code_block(lines)
    max_tokens = 55
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=max_tokens, provenance_token_reserve=12)
    for p in parts:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _WORD_TOK(full_text) <= max_tokens


def test_provenance_header_is_included_in_the_token_budget():
    lines = [f"x{i} = compute_something_of_moderate_length({i})" for i in range(20)]
    block = _code_block(lines)
    small_reserve = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=60, provenance_token_reserve=2)
    large_reserve = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=60, provenance_token_reserve=40)
    # Reserving more room for the header leaves less room for content, so it
    # can only produce the same number of parts or more -- never fewer.
    assert len(large_reserve) >= len(small_reserve)
    for p in large_reserve:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _WORD_TOK(full_text) <= 60
        assert _WORD_TOK(p.context_header) > 0


def test_parent_block_id_is_preserved_on_every_part():
    lines = [f"line {i} filler filler filler filler filler" for i in range(20)]
    block = _code_block(lines, block_id="b0271")
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10)
    assert len(parts) > 1
    for p in parts:
        assert p.block.parent_id == "b0271"
        assert p.provenance["parent_block_id"] == "b0271"
        assert p.provenance["block_part_id"] == p.block.block_id
        assert p.block.block_id.startswith("b0271-part-")


def test_line_ranges_are_preserved_and_contiguous():
    lines = [f"line {i} filler filler filler filler filler" for i in range(20)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10)
    assert parts[0].line_start == 1
    assert parts[-1].line_end == len(lines)
    for prev, cur in zip(parts, parts[1:]):
        assert cur.line_start == prev.line_end + 1


def test_controlled_overlap_is_recorded_correctly():
    lines = [f"line {i} filler filler filler filler filler" for i in range(20)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(
        block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10, overlap_tokens=15,
    )
    assert len(parts) > 1
    for prev, cur in zip(parts, parts[1:]):
        if cur.overlap_line_start is not None:
            assert cur.overlap_line_end == prev.line_end
            assert cur.overlap_line_start <= cur.overlap_line_end
            overlapped = lines[cur.overlap_line_start - 1: cur.overlap_line_end]
            assert all(line in cur.block.code for line in overlapped)


def test_parts_reconstruct_to_the_original_normalized_block():
    lines = [f"line {i} filler filler filler filler filler" for i in range(23)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(
        block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10, overlap_tokens=10,
    )
    recon = verify_reconstruction(lines, parts)
    assert recon["reconstructs_to_original"] is True
    assert recon["missing_lines"] == []
    assert recon["duplicated_non_overlap_lines"] == []


def test_no_non_overlap_line_is_lost():
    lines = [f"line {i} filler filler filler filler filler" for i in range(37)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=45, provenance_token_reserve=10)
    covered = set()
    for p in parts:
        covered.update(range(p.line_start, p.line_end + 1))
    assert covered == set(range(1, len(lines) + 1))


def test_no_non_overlap_line_is_duplicated():
    lines = [f"line {i} filler filler filler filler filler" for i in range(37)]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(
        block, tokenizer=_WORD_TOK, max_tokens=45, provenance_token_reserve=10, overlap_tokens=12,
    )
    seen = []
    for p in parts:
        seen.extend(range(p.line_start, p.line_end + 1))
    assert len(seen) == len(set(seen))


def test_single_overlong_line_forces_subdivision_without_truncation():
    long_line = " ".join(f"word{i}" for i in range(100))
    lines = [long_line]
    block = _code_block(lines)
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=30, provenance_token_reserve=5)
    assert len(parts) >= 1
    assert parts[0].forced_split is True
    # No word from the original line is lost, and none are duplicated.
    rebuilt_words = " ".join(p.block.code for p in parts).split()
    assert rebuilt_words == long_line.split()
    for p in parts:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _WORD_TOK(full_text) <= 30


def test_table_split_repeats_the_header_in_every_part():
    header = ["Name", "Value", "Description"]
    rows = [[f"row{i}", str(i), f"description of row {i} with some extra words"] for i in range(20)]
    block = TableBlock(block_id="tbl1", order=0, source_format="pdf", header=header, rows=rows)
    # max_tokens=40 (word-count budget) fit under the old packer's estimate
    # (raw " | ".join(header)/" | ".join(row) with no separator row, no
    # leading/trailing pipes) but does not fit the real serialized part
    # (render_table's header + separator + padded/piped rows) -- 115 is
    # sized against the real rendering so this still exercises "several
    # rows comfortably packed per part, several parts, never a forced row
    # split" under the correct accounting.
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=115, provenance_token_reserve=10)
    assert len(parts) > 1
    assert not any(p.forced_split for p in parts)
    for p in parts:
        assert p.block.header == header
    # Rows are never split across parts (each row here is well under budget).
    all_rows = [tuple(r) for p in parts for r in p.block.rows]
    assert all_rows == [tuple(r) for r in rows]


def test_under_budget_table_passes_through_unsplit():
    # Gating (chunker.py's _pack_blocks) only ever calls the splitter once a
    # block is measured as oversized against the real renderer/tokenizer --
    # a table that comfortably fits never reaches split_oversized_atomic_block
    # at all. Exercised end to end through the real chunker with the real
    # tokenizer, not just asserted about the splitter in isolation.
    text = "# T\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
    doc, _log = parse_markdown(text, source_path="x.md")
    normalize_document(doc)
    config = ChunkingConfig(hard_max_words=500, target_words=400)
    result = MarkdownChunker().chunk(
        doc, "x.md", config, document_id="d1", token_counter=_WORD_TOK, max_sequence_tokens=512,
    )
    assert result.atomic_blocks_split == 0
    assert any("| A | B |" in c.text for c in result.chunks)


@requires_real_tokenizer
def test_no_table_part_exceeds_the_real_tokenizer_limit():
    header = ["Part No.", "Spec", "Notes"]
    rows = [
        [f"P-{i:04d}", f"Torque {i} Nm rated class A", f"Notes for row {i}: inspect and record measurements"]
        for i in range(30)
    ]
    block = _table_block(header, rows, block_id="rtbl")
    max_tokens = 150
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=20)
    assert len(parts) > 1
    for p in parts:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _REAL_TOK(full_text) <= max_tokens


@requires_real_tokenizer
def test_escaped_pipe_cells_do_not_undercount_the_real_render():
    # Reproduces the diagnosed bug directly: literal "|" in cell content
    # costs more once _escape_table_cell() escapes it for the real render
    # (render_table) than the old packer's raw, un-escaped tokenization
    # assumed -- this table would have raised AtomicSplitError under the
    # old packer (confirmed via a standalone repro against this same
    # tokenizer before the fix).
    header = ["Part No.", "Spec", "Notes"]
    rows = [
        [f"P-{i:04d}", f"Torque {i} Nm | rated | class A", f"Notes {i}: inspect | verify | record"]
        for i in range(40)
    ]
    block = _table_block(header, rows, block_id="pipe-tbl")
    max_tokens = 120
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=20)
    assert len(parts) > 1
    for p in parts:
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _REAL_TOK(full_text) <= max_tokens
    all_rows = [tuple(r) for p in parts for r in p.block.rows]
    assert all_rows == [tuple(r) for r in rows]


@requires_real_tokenizer
def test_table_parts_round_trip_through_the_real_parser():
    # Cell content here is pipe/newline-free by design -- this test checks
    # structural row/column re-parse fidelity of a split part, which is
    # independent of md_parser.py's separate (pre-existing, out of scope
    # for this change) handling of "\|" un-escaping on re-parse.
    header = ["Name", "Value", "Description"]
    rows = [
        [f"row{i}", str(i), f"description of row {i} with several extra padding words here for length"]
        for i in range(25)
    ]
    block = _table_block(header, rows, block_id="rt")
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=140, provenance_token_reserve=15)
    assert len(parts) > 1
    for p in parts:
        md = render_block(p.block)
        reparsed_doc, _log = parse_markdown(md, source_path="x.md")
        table_blocks = [b for b in reparsed_doc.blocks if isinstance(b, TableBlock)]
        assert len(table_blocks) == 1
        assert table_blocks[0].header == p.block.header
        assert table_blocks[0].rows == p.block.rows


@requires_real_tokenizer
def test_forced_row_column_group_subdivision_preserves_the_key_cell():
    header = ["ID", "ColA", "ColB", "ColC", "ColD"]
    padding = " ".join(f"filler{i}" for i in range(40))
    oversized_row = ["KEY1", f"alpha {padding}", f"beta {padding}", f"gamma {padding}", f"delta {padding}"]
    rows = [["r0", "a", "b", "c", "d"], oversized_row, ["r2", "a2", "b2", "c2", "d2"]]
    block = _table_block(header, rows, block_id="colgrp")
    max_tokens = 130
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=15)
    forced_parts = [p for p in parts if p.forced_split]
    assert len(forced_parts) > 1  # the oversized row required more than one fragment
    for p in forced_parts:
        assert p.block.header[0] == "ID"
        assert p.block.rows[0][0] == "KEY1"  # key cell repeated on every fragment
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _REAL_TOK(full_text) <= max_tokens
    # Every non-key column's content is represented somewhere across the
    # fragments -- nothing from the oversized row is silently dropped.
    covered_cols = set()
    for p in forced_parts:
        covered_cols.update(p.block.header[1:])
    assert covered_cols == {"ColA", "ColB", "ColC", "ColD"}
    # Untouched rows pass through verbatim, still with the full header.
    normal_parts = [p for p in parts if not p.forced_split]
    assert any(p.block.rows == [["r0", "a", "b", "c", "d"]] for p in normal_parts) or any(
        ["r0", "a", "b", "c", "d"] in p.block.rows for p in normal_parts
    )


@requires_real_tokenizer
def test_single_oversized_cell_falls_back_to_text_splitting():
    header = ["ID", "Notes"]
    huge_note = " ".join(f"word{i}" for i in range(300))
    block = _table_block(header, [["KEYX", huge_note]], block_id="cellsplit")
    max_tokens = 60
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=10)
    assert len(parts) > 1
    assert all(p.forced_split for p in parts)
    for p in parts:
        assert p.block.header == ["ID", "Notes"]
        assert p.block.rows[0][0] == "KEYX"
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _REAL_TOK(full_text) <= max_tokens
    # No word from the oversized cell is lost or duplicated across pieces.
    rebuilt = " ".join(p.block.rows[0][1] for p in parts)
    assert rebuilt.split() == huge_note.split()


@requires_real_tokenizer
def test_sentence_length_column_headers_subdivide_with_per_fragment_budgets():
    # Regression for the observed Azure-docs production failure: column-header
    # text is sentence-length, so the old single worst-case ALL-columns sample
    # header ("... columns col1-col2-...-colN, forced split") consumed the
    # entire budget before any cell was measured -- frag_budget collapsed to 1
    # and ladder rule (c) raised AtomicSplitError ("its key cell alone exceeds
    # the available budget (1 tokens)"), failing the whole document in strict
    # mode. With per-fragment budget accounting each fragment is measured
    # against its OWN columns descriptor, so the row now subdivides.
    long_header = lambda label: label + " " + " ".join(  # noqa: E731
        f"descriptor{i}" for i in range(15)
    )
    header = ["Setting"] + [long_header(f"Column{c}") for c in range(6)]
    padding = " ".join(f"value{i}" for i in range(30))
    row = ["enable-feature"] + [f"{c} {padding}" for c in range(6)]
    block = _table_block(header, [row], block_id="aztbl")
    max_tokens = 180
    parts = split_oversized_atomic_block(block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=20)
    # The one oversized row explodes into per-column fragments, each with the
    # key cell repeated and each within the real budget.
    assert len(parts) >= len(header) - 1
    assert all(p.forced_split for p in parts)
    for p in parts:
        assert p.block.rows[0][0] == "enable-feature"
        full_text = p.context_header + "\n\n" + render_block(p.block)
        assert _REAL_TOK(full_text) <= max_tokens
    covered_cols = set()
    for p in parts:
        covered_cols.update(p.block.header[1:])
    assert covered_cols == set(header[1:])


@requires_real_tokenizer
def test_oversized_row_policy_error_names_the_row():
    header = ["ID", "Notes"]
    huge_note = " ".join(f"word{i}" for i in range(300))
    block = _table_block(header, [["r0", "short"], ["KEYX", huge_note]], block_id="errpolicy")
    with pytest.raises(AtomicSplitError) as exc_info:
        split_oversized_atomic_block(
            block, tokenizer=_REAL_TOK, max_tokens=60, provenance_token_reserve=10,
            oversized_row_policy="error",
        )
    message = str(exc_info.value)
    assert "errpolicy" in message
    assert "row 2" in message  # names the specific (1-based) row


def _random_table(rng: random.Random):
    n_cols = rng.randint(1, 5)
    header = [f"Col{c}" for c in range(n_cols)]
    n_rows = rng.randint(1, 30)
    rows = []
    for _ in range(n_rows):
        row = []
        for _c in range(n_cols):
            n_words = rng.randint(1, 6)
            cell = " ".join(f"w{rng.randint(0, 999)}" for _ in range(n_words))
            if rng.random() < 0.15:
                cell += " | literal pipe | here"
            if rng.random() < 0.1:
                cell = "line one\nline two " + cell
            row.append(cell)
        rows.append(row)
    return header, rows


@requires_real_tokenizer
def test_random_tables_never_exceed_budget_and_preserve_row_order():
    # Property test: randomized row/column counts and cell densities
    # (including literal "|" and "\n" in cells), asserting the invariants
    # that matter regardless of table shape -- every emitted part fits the
    # real per-part budget, no part is empty, parts are numbered in order,
    # and (in the common case where nothing needed the forced-row ladder)
    # every row survives exactly once, in original order.
    rng = random.Random(20260903)
    for trial in range(30):
        header, rows = _random_table(rng)
        block = _table_block(header, rows, block_id=f"rand{trial}")
        max_tokens = rng.randint(80, 250)
        reserve = rng.randint(10, 30)
        try:
            parts = split_oversized_atomic_block(
                block, tokenizer=_REAL_TOK, max_tokens=max_tokens, provenance_token_reserve=reserve,
            )
        except AtomicSplitError:
            # A budget too small to even hold one row's key cell is a
            # legitimate hard stop (ladder rule (c)), not a property failure.
            continue

        assert len(parts) > 0
        assert [p.part_index for p in parts] == list(range(1, len(parts) + 1))
        assert all(p.part_count == len(parts) for p in parts)
        for p in parts:
            assert len(p.block.rows) >= 1  # no empty parts
            full_text = p.context_header + "\n\n" + render_block(p.block)
            assert _REAL_TOK(full_text) <= max_tokens, (trial, max_tokens, p.block.block_id)

        if not any(p.forced_split for p in parts):
            all_rows = [tuple(r) for p in parts for r in p.block.rows]
            assert all_rows == [tuple(r) for r in rows]


def test_list_split_preserves_numbering_context():
    items = [ListItemNode(text=f"item number {i} with a few extra padding words") for i in range(25)]
    block = ListBlock(block_id="lst1", order=0, source_format="pdf", ordered=True, items=items)
    parts = split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=40, provenance_token_reserve=10)
    assert len(parts) > 1
    for p in parts[1:]:
        assert "continuing from item" in p.context_header
    # Original item order/content preserved across parts (minus overlap duplication).
    own_texts = []
    for p in parts:
        skip = 1 if p.overlap_line_start else 0
        own_texts.extend(it.text for it in p.block.items[skip:])
    assert own_texts == [it.text for it in items]


def test_preview_and_index_paths_produce_identical_boundaries_for_the_same_tokenizer():
    text = "# T\n\n```\n" + "\n".join(f"line {i} filler filler filler filler" for i in range(30)) + "\n```\n"
    doc1, _ = parse_markdown(text, source_path="x.md")
    normalize_document(doc1)
    doc2, _ = parse_markdown(text, source_path="x.md")
    normalize_document(doc2)

    config = ChunkingConfig(hard_max_words=40, target_words=30)
    r1 = MarkdownChunker().chunk(doc1, "x.md", config, document_id="d1", token_counter=_WORD_TOK, max_sequence_tokens=60)
    r2 = MarkdownChunker().chunk(doc2, "x.md", config, document_id="d1", token_counter=_WORD_TOK, max_sequence_tokens=60)

    assert len(r1.chunks) > 1
    assert [c.text for c in r1.chunks] == [c.text for c in r2.chunks]
    assert [c.source_block_ids for c in r1.chunks] == [c.source_block_ids for c in r2.chunks]


def test_atomic_split_error_raised_when_budget_too_small_to_help():
    lines = [f"line {i} filler filler filler filler filler" for i in range(5)]
    block = _code_block(lines)
    try:
        split_oversized_atomic_block(block, tokenizer=_WORD_TOK, max_tokens=3, provenance_token_reserve=40)
    except AtomicSplitError:
        pass
    else:
        raise AssertionError("expected AtomicSplitError for a budget too small to hold even one part")
