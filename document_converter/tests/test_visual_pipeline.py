"""
Unit tests for the VISUAL_ANALYSIS pipeline: region extraction/filtering
(visual_extraction.py), text association (visual_association.py), VLM
response validation/retrieval-text generation (visual_analysis.py), the
filesystem analysis cache (visual_cache.py), and index-schema record
shape/content_kind separation (index_schema.py). No live Ollama/embedding
service required -- see artifacts/visual_pipeline_validation/ for the real-
document, real-model acceptance evidence these unit tests don't replace.
"""
from __future__ import annotations

import json
import os

import pymupdf
import pytest

from document_converter.config import VisualConfig
from document_converter.chunker import ChunkRecord
from document_converter import visual_extraction, visual_association, visual_analysis, visual_cache, gpu_lifecycle
from document_converter.index_schema import (
    build_visual_record, build_index_payload, load_index, CONTENT_KIND_MAIN_TEXT, CONTENT_KIND_VISUAL_ANALYSIS,
)
from document_converter.tests.conftest import FakeEmbeddingClient


# ---------------------------------------------------------------------- #
# visual_extraction.py -- built against a small synthetic PDF (drawn with
# pymupdf itself) rather than a real corpus file, so this test is fast,
# hermetic, and asserts exact pixel geometry; real-corpus extraction is
# covered by the acceptance run, not duplicated here.

def _make_synthetic_pdf(path: str):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)  # US Letter, points
    # A small decorative icon (below both size thresholds) near the top.
    icon_rect = pymupdf.Rect(40, 40, 60, 60)
    icon_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    icon_pix.set_rect(icon_pix.irect, (200, 30, 30))
    page.insert_image(icon_rect, pixmap=icon_pix)
    # A large, eligible "chart-sized" image in the body.
    chart_rect = pymupdf.Rect(100, 200, 500, 500)
    chart_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300))
    chart_pix.set_rect(chart_pix.irect, (30, 100, 200))
    page.insert_image(chart_rect, pixmap=chart_pix)
    # A thin horizontal separator line (full width, 1pt tall).
    sep_rect = pymupdf.Rect(40, 550, 570, 551)
    sep_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 530, 1))
    sep_pix.set_rect(sep_pix.irect, (0, 0, 0))
    page.insert_image(sep_rect, pixmap=sep_pix)
    doc.save(path)
    doc.close()


@pytest.fixture
def synthetic_pdf(tmp_path):
    path = str(tmp_path / "synthetic.pdf")
    _make_synthetic_pdf(path)
    return path


def test_extract_visual_regions_finds_the_eligible_chart_only(synthetic_pdf):
    regions, skipped = visual_extraction.extract_visual_regions(synthetic_pdf, VisualConfig())
    assert len(regions) == 1
    assert regions[0].page == 1
    assert regions[0].width_pt == pytest.approx(400, abs=1)
    assert regions[0].height_pt == pytest.approx(300, abs=1)
    assert len(regions[0].image_bytes) > 0
    assert len(regions[0].image_hash) == 64  # sha256 hex


def test_extract_visual_regions_skips_icon_and_separator(synthetic_pdf):
    # Both the small icon and the thin separator line are skipped as
    # "too small" -- a separator's short dimension is caught by
    # `min_dimension_pt` the same way a small icon's full bbox is (see
    # VisualSkipReason.TOO_SMALL's docstring).
    regions, skipped = visual_extraction.extract_visual_regions(synthetic_pdf, VisualConfig())
    reasons = {s["reason"] for s in skipped}
    assert len(skipped) == 2
    assert reasons == {visual_extraction.VisualSkipReason.TOO_SMALL}


def test_extract_visual_regions_non_pdf_returns_empty(tmp_path):
    txt_path = tmp_path / "not_a_pdf.txt"
    txt_path.write_text("hello")
    regions, skipped = visual_extraction.extract_visual_regions(str(txt_path), VisualConfig())
    assert regions == []
    assert skipped == []


def test_extract_visual_regions_respects_max_visuals_budget(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=1600)
    for i in range(5):
        rect = pymupdf.Rect(50, 50 + i * 300, 450, 300 + i * 300)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 250))
        pix.set_rect(pix.irect, (10 * i, 50, 150))
        page.insert_image(rect, pixmap=pix)
    path = str(tmp_path / "many.pdf")
    doc.save(path)
    doc.close()

    cfg = VisualConfig(max_visuals_per_document=2)
    regions, skipped = visual_extraction.extract_visual_regions(path, cfg)
    assert len(regions) == 2
    assert any(s["reason"] == visual_extraction.VisualSkipReason.BUDGET for s in skipped)


def test_dedupe_nested_keeps_largest():
    boxes = [
        {"bbox": [10, 10, 200, 200]},
        {"bbox": [12, 12, 198, 198]},  # nested inside the first
        {"bbox": [300, 300, 400, 400]},  # unrelated, kept separately
    ]
    kept = visual_extraction._dedupe_nested(boxes)
    assert len(kept) == 2
    assert [10, 10, 200, 200] in [k["bbox"] for k in kept]


def test_save_crop_is_content_addressed_and_idempotent(tmp_path, synthetic_pdf):
    regions, _ = visual_extraction.extract_visual_regions(synthetic_pdf, VisualConfig())
    index_dir = str(tmp_path / "index")
    p1 = visual_extraction.save_crop(index_dir, regions[0])
    p2 = visual_extraction.save_crop(index_dir, regions[0])
    assert p1 == p2
    assert os.path.exists(p1)
    assert os.path.basename(p1) == f"{regions[0].image_hash}.png"


# ---------------------------------------------------------------------- #
# vector-drawn visual detection (2026-08-13 correction phase) -- synthetic
# fixtures built with pymupdf itself, mirroring the geometric shapes real
# acceptance evidence was gathered against (see RAG_BUILD_SOURCE_OF_TRUTH.md
# section 22 for the real-PDF evidence these unit tests don't replace):
# a bar chart (axis lines + rects), a decorative rule, a table grid, and a
# synthetic "vector-rendered body text" page (many small, roughly-square
# blob-shaped curves covering most of the page, no embedded text layer --
# reproducing the real false-positive found on
# Learning-Bayesian-Networks-from-Big-Data-with-Greedy-Search.pdf).

def _make_vector_bar_chart_pdf(path: str):
    # Axes and bars are drawn with separate finish()/commit() calls, which
    # is how real PDF-generating chart libraries normally produce them too
    # (different stroke/fill styles -- an unfilled axis stroke vs. filled
    # bars -- are different path objects; PyMuPDF's own `get_drawings()`
    # groups by exactly this "distinct path/style object" granularity, not
    # by visual concept), rather than one artificial bundled path mixing
    # both element types.
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    axes = page.new_shape()
    axes.draw_line((100, 500), (100, 200))  # y-axis
    axes.draw_line((100, 500), (400, 500))  # x-axis
    axes.finish(width=1.5, color=(0, 0, 0))
    axes.commit()
    bars = page.new_shape()
    # 10pt gaps (axis-to-first-bar and bar-to-bar) -- comfortably within
    # the default `vector_group_gap_pt` grouping distance, matching how
    # tightly real chart-generator output was observed to be spaced on
    # the real corpus PDFs this phase's acceptance evidence is built on.
    for i, h in enumerate([50, 120, 80, 150, 60]):
        x0 = 110 + i * 40
        bars.draw_rect(pymupdf.Rect(x0, 500 - h, x0 + 30, 500))
    bars.finish(width=1.0, color=(0, 0, 0), fill=(0.3, 0.3, 0.8))
    bars.commit()
    doc.save(path)
    doc.close()


def _make_vector_decorative_pdf(path: str):
    """A page with only a horizontal separator rule and a small ruled
    table grid -- both should be rejected (too small / table-like), never
    sent to the VLM."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    sep = page.new_shape()
    sep.draw_line((40, 700), (570, 700))
    sep.finish(width=1)
    sep.commit()

    grid = page.new_shape()
    for r in range(6):
        grid.draw_line((300, 100 + r * 15), (580, 100 + r * 15))
    for c in range(5):
        grid.draw_line((300 + c * 56, 100), (300 + c * 56, 175))
    grid.finish(width=0.5)
    grid.commit()
    doc.save(path)
    doc.close()


def _make_vector_text_like_pdf(path: str):
    """Simulates a page whose body text was rendered as vector glyph
    outlines with no extractable text layer (the real false-positive
    pattern found on a real corpus PDF -- see
    `VisualConfig.vector_min_thin_primitive_fraction`'s docstring): many
    small, roughly-square (low-aspect-ratio) filled curve blobs tiled in
    text-line rows across most of the page, zero real `get_text()` output."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    for row in range(40):
        y = 60 + row * 16
        for col in range(60):
            x = 60 + col * 8
            # A small roughly-square blob (curve-ish via a tiny filled
            # rect standing in for a glyph outline) -- aspect ratio ~1,
            # unlike a chart's thin lines/bars.
            shape.draw_rect(pymupdf.Rect(x, y, x + 5, y + 7))
    shape.finish(width=0, fill=(0, 0, 0))
    shape.commit()
    doc.save(path)
    doc.close()


def test_extract_visual_regions_detects_vector_bar_chart(tmp_path):
    path = str(tmp_path / "vector_chart.pdf")
    _make_vector_bar_chart_pdf(path)
    regions, skipped = visual_extraction.extract_visual_regions(path, VisualConfig())
    assert len(regions) == 1
    assert regions[0].source == "vector"
    assert regions[0].page == 1
    assert regions[0].vector_primitive_count is not None and regions[0].vector_primitive_count >= 7
    assert len(regions[0].image_bytes) > 0


def test_extract_visual_regions_rejects_decorative_vector_shapes(tmp_path):
    path = str(tmp_path / "vector_decorative.pdf")
    _make_vector_decorative_pdf(path)
    regions, skipped = visual_extraction.extract_visual_regions(path, VisualConfig())
    assert regions == []
    reasons = {s["reason"] for s in skipped}
    # The separator is too small; the table grid is either caught by the
    # too-small check or explicitly recognized as table-like -- either way
    # neither reaches the VLM.
    assert reasons.issubset({
        visual_extraction.VisualSkipReason.VECTOR_TOO_SMALL,
        visual_extraction.VisualSkipReason.VECTOR_TOO_SPARSE,
        visual_extraction.VisualSkipReason.VECTOR_TABLE_LIKE,
        visual_extraction.VisualSkipReason.VECTOR_GRID_LIKE,
        visual_extraction.VisualSkipReason.VECTOR_LOW_DIVERSITY,
    })


def test_extract_visual_regions_rejects_vector_rendered_body_text(tmp_path):
    # Regression test for a real false positive (RAG_BUILD_SOURCE_OF_TRUTH.md
    # section 22): a page with no extractable text layer whose body text was
    # rendered as thousands of small vector glyph outlines must not be
    # treated as a chart/diagram.
    path = str(tmp_path / "vector_text_like.pdf")
    _make_vector_text_like_pdf(path)
    regions, skipped = visual_extraction.extract_visual_regions(path, VisualConfig())
    assert regions == []
    assert any(s["reason"] == visual_extraction.VisualSkipReason.VECTOR_TEXT_LIKE for s in skipped)


def test_is_thin_primitive_distinguishes_lines_from_blobs():
    # A near-horizontal line (chart axis/gridline shape) is thin.
    assert visual_extraction._is_thin_primitive([0.0, 0.0, 300.0, 0.5], aspect_ratio=4.0)
    # A roughly-square blob (glyph/node shape) is not.
    assert not visual_extraction._is_thin_primitive([0.0, 0.0, 6.0, 7.0], aspect_ratio=4.0)


def test_dedupe_cross_source_prefers_raster_on_overlap():
    raster = {"bbox": [100.0, 100.0, 400.0, 400.0], "source": "raster"}
    vector = {"bbox": [110.0, 110.0, 390.0, 390.0], "source": "vector", "vector_primitive_count": 20}
    kept, dropped = visual_extraction._dedupe_cross_source([raster, vector], iou_threshold=0.15)
    assert kept == [raster]
    assert dropped[0]["reason"] == visual_extraction.VisualSkipReason.DUPLICATE_OF_RASTER


def test_dedupe_cross_source_keeps_non_overlapping_vector():
    raster = {"bbox": [100.0, 100.0, 200.0, 200.0], "source": "raster"}
    vector = {"bbox": [300.0, 300.0, 400.0, 400.0], "source": "vector", "vector_primitive_count": 20}
    kept, dropped = visual_extraction._dedupe_cross_source([raster, vector], iou_threshold=0.15)
    assert raster in kept and vector in kept
    assert dropped == []


def test_group_drawings_by_proximity_merges_nearby_and_separates_distant():
    bboxes = [
        [100.0, 100.0, 110.0, 110.0],
        [115.0, 100.0, 125.0, 110.0],  # 5pt gap from the first -- merges under a 12pt gap
        [500.0, 500.0, 510.0, 510.0],  # far away -- separate cluster
    ]
    groups = visual_extraction._group_drawings_by_proximity(bboxes, gap_pt=12.0)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


# ---------------------------------------------------------------------- #
# visual_association.py

def _chunk(chunk_id, chunk_index, header_path, pages):
    return ChunkRecord(
        chunk_id=chunk_id, chunk_index=chunk_index, header_path=header_path, heading=header_path[-1],
        provenance="[doc.pdf > " + " > ".join(header_path) + "]", text="text", body="text",
        word_count=1, token_count=1, source_block_ids=[], source_pages=pages,
    )


def test_associate_visual_prefers_same_page_chunks():
    chunks = [
        _chunk("c1", 0, ["Intro"], [1]),
        _chunk("c2", 1, ["Results"], [2]),
        _chunk("c3", 2, ["Results"], [2, 3]),
        _chunk("c4", 3, ["Conclusion"], [4]),
    ]
    assoc = visual_association.associate_visual(page=2, chunks=chunks)
    assert assoc.linked_chunk_ids == ["c2", "c3"]
    assert assoc.heading_path == ["Results"]


def test_associate_visual_falls_back_to_nearest_neighbors_when_no_same_page_chunk():
    chunks = [
        _chunk("c1", 0, ["Intro"], [1]),
        _chunk("c2", 1, ["Results"], [3]),  # nothing on page 2
    ]
    assoc = visual_association.associate_visual(page=2, chunks=chunks)
    assert assoc.linked_chunk_ids == ["c1", "c2"]


def test_associate_visual_caps_linked_chunks():
    chunks = [_chunk(f"c{i}", i, ["Results"], [2]) for i in range(10)]
    assoc = visual_association.associate_visual(page=2, chunks=chunks)
    assert len(assoc.linked_chunk_ids) == visual_association.MAX_LINKED_CHUNKS


def test_associate_visual_empty_chunks_is_safe():
    assoc = visual_association.associate_visual(page=1, chunks=[])
    assert assoc.linked_chunk_ids == []
    assert assoc.heading_path == []


# ---------------------------------------------------------------------- #
# visual_analysis.py -- schema validation / grounding contract enforcement

VALID_RESPONSE = json.dumps({
    "visual_type": "bar_chart",
    "title": "Memory Usage",
    "summary": "Bar chart comparing memory usage before/after optimization.",
    "axes": {"x": {"label": "Memory Usage", "units": "%"}, "y": {"label": "Process", "units": ""}},
    "legend": ["Before Optimization", "After Optimization"],
    "data_points": [{"label": "Process 0", "value": "~85% before, ~70% after"}],
    "relationships": ["Every process improved after optimization."],
    "trend": "Decreasing memory usage after optimization.",
    "key_takeaway": "Optimization reduced memory usage across all processes.",
    "uncertainties": [],
})


def test_parse_and_validate_accepts_well_formed_response():
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    assert analysis.visual_type == "bar_chart"
    assert analysis.title == "Memory Usage"
    assert analysis.axes["x"]["units"] == "%"
    assert len(analysis.data_points) == 1


def test_parse_and_validate_strips_markdown_code_fences():
    fenced = "```json\n" + VALID_RESPONSE + "\n```"
    analysis = visual_analysis.parse_and_validate(fenced)
    assert analysis.visual_type == "bar_chart"


def test_parse_and_validate_salvages_json_wrapped_in_prose():
    wrapped = "Here is the analysis:\n" + VALID_RESPONSE + "\nHope that helps!"
    analysis = visual_analysis.parse_and_validate(wrapped)
    assert analysis.visual_type == "bar_chart"


def test_parse_and_validate_rejects_empty_response():
    with pytest.raises(visual_analysis.VLMInvalidResponseError):
        visual_analysis.parse_and_validate("")


def test_parse_and_validate_rejects_non_json():
    with pytest.raises(visual_analysis.VLMInvalidResponseError):
        visual_analysis.parse_and_validate("I cannot analyze this image.")


def test_parse_and_validate_rejects_missing_summary():
    bad = json.dumps({"visual_type": "bar_chart", "title": "x"})
    with pytest.raises(visual_analysis.VLMInvalidResponseError):
        visual_analysis.parse_and_validate(bad)


def test_parse_and_validate_downgrades_unknown_visual_type_to_other():
    bad = json.dumps({"visual_type": "hologram", "summary": "a chart"})
    analysis = visual_analysis.parse_and_validate(bad)
    assert analysis.visual_type == "other"


def test_parse_and_validate_coerces_malformed_optional_fields():
    bad = json.dumps({
        "visual_type": "bar_chart", "summary": "ok",
        "legend": "not a list", "data_points": "also not a list", "axes": "wrong type",
    })
    analysis = visual_analysis.parse_and_validate(bad)
    assert analysis.legend == []
    assert analysis.data_points == []
    assert analysis.axes == {"x": {"label": "", "units": ""}, "y": {"label": "", "units": ""}}


def test_build_retrieval_text_bounds_length_for_densely_connected_diagrams():
    # Regression test for a real failure observed during acceptance testing
    # (RAG_BUILD_SOURCE_OF_TRUTH.md change log): a variable-dependency
    # diagram with few relationship sentences that were each individually
    # very long produced retrieval text exceeding BGE's 512-token limit,
    # even though item-count alone looked fine.
    variables = ["Day", "Month", "Hour", "Season", "Year", "Latitude", "Longitude",
                 "Altitude", "Region", "Zone", "Type", "CVD60", "tp", "ssr"]
    long_relationships = [
        f"{v} influences " + ", ".join(x for x in variables if x != v) + "."
        for v in variables
    ]
    raw = json.dumps({
        "visual_type": "diagram", "summary": "A dependency diagram.",
        "relationships": long_relationships,
    })
    analysis = visual_analysis.parse_and_validate(raw)
    text = visual_analysis.build_retrieval_text(analysis, heading_path=["Section"], page=11)
    assert len(text.split()) <= 260  # comfortably under BGE's 512-token ceiling
    # Header/summary lines (needed for citation/identification) survive.
    assert "Diagram" in text
    assert "Page: 11." in text


def test_build_retrieval_text_contains_grounded_facts_not_filler():
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    text = visual_analysis.build_retrieval_text(analysis, heading_path=["Results", "Memory Usage"], page=6)
    assert "Bar chart" in text
    assert "Memory Usage" in text
    assert "Page: 6." in text
    assert "Results > Memory Usage" in text
    assert "colorful" not in text.lower()


# ---------------------------------------------------------------------- #
# visual_cache.py

def test_cache_round_trip(tmp_path):
    index_dir = str(tmp_path / "index")
    assert visual_cache.get_cached_analysis(index_dir, "abc123", "qwen3-vl:8b-instruct-q4_K_M", "1") is None
    visual_cache.write_cached_analysis(
        index_dir, "abc123", "qwen3-vl:8b-instruct-q4_K_M", "1",
        raw_response="{}", analysis={"visual_type": "bar_chart", "summary": "x"}, retrieval_text="Visual type: Bar chart.",
    )
    cached = visual_cache.get_cached_analysis(index_dir, "abc123", "qwen3-vl:8b-instruct-q4_K_M", "1")
    assert cached is not None
    assert cached["analysis"]["visual_type"] == "bar_chart"
    assert cached["retrieval_text"] == "Visual type: Bar chart."


def test_cache_miss_on_different_model():
    pass  # covered by test_cache_invalidates_on_version_or_model_change below


def test_cache_invalidates_on_version_or_model_change(tmp_path):
    index_dir = str(tmp_path / "index")
    visual_cache.write_cached_analysis(
        index_dir, "abc123", "qwen3-vl:8b-instruct-q4_K_M", "1", "{}", {"visual_type": "bar_chart", "summary": "x"}, "text",
    )
    # Different analysis_version -> miss, even though the image is unchanged.
    assert visual_cache.get_cached_analysis(index_dir, "abc123", "qwen3-vl:8b-instruct-q4_K_M", "2") is None
    # Different model -> miss.
    assert visual_cache.get_cached_analysis(index_dir, "abc123", "some-other-vlm:latest", "1") is None
    # Different image hash -> miss.
    assert visual_cache.get_cached_analysis(index_dir, "def456", "qwen3-vl:8b-instruct-q4_K_M", "1") is None


def test_cache_miss_on_corrupt_file(tmp_path):
    index_dir = str(tmp_path / "index")
    os.makedirs(os.path.join(index_dir, "_visuals"), exist_ok=True)
    bad_path = os.path.join(index_dir, "_visuals", "abc123__model__v1.json")
    with open(bad_path, "w") as f:
        f.write("{not valid json")
    assert visual_cache.get_cached_analysis(index_dir, "abc123", "model", "1") is None


# ---------------------------------------------------------------------- #
# index_schema.py -- content_kind separation

def test_build_visual_record_has_visual_content_kind_and_required_fields():
    rec = build_visual_record(
        visual_id="v-doc123-abcdef", document_id="doc123", source_path="/x/doc.pdf",
        source_hash="hash", page=6, bbox=[1.0, 2.0, 3.0, 4.0], visual_type="bar_chart",
        heading_path=["Results"], linked_chunk_ids=["c1", "c2"], image_path="/x/_visuals/abc.png",
        image_hash="abc", analysis_model="qwen3-vl:8b-instruct-q4_K_M", analysis_model_digest="deadbeef",
        analysis_version="1", retrieval_text="Visual type: Bar chart.", analysis={"visual_type": "bar_chart"},
        document_order=600001, embedding=[0.1, 0.2],
    )
    assert rec["content_kind"] == CONTENT_KIND_VISUAL_ANALYSIS
    assert rec["visual_id"] == "v-doc123-abcdef"
    assert rec["page"] == 6
    assert rec["linked_chunk_ids"] == ["c1", "c2"]
    assert rec["embedding"] == [0.1, 0.2]
    assert rec["text"] == "Visual type: Bar chart."
    assert rec["analysis_model_digest"] == "deadbeef"


def test_build_index_payload_main_text_chunks_default_content_kind(tmp_path):
    class FakeEmbeddingInfo:
        provider = "sentence-transformers-service"
        model = "BAAI/bge-large-en-v1.5"
        revision = "rev"
        dimension = 2
        max_sequence_tokens = 512
        normalized_embeddings = True
        query_prompt_name = None
        tokenizer_id = "BAAI/bge-large-en-v1.5"

    chunk = ChunkRecord(
        chunk_id="c1", chunk_index=0, header_path=["Intro"], heading="Intro",
        provenance="[doc.pdf > Intro]", text="hello", body="hello", word_count=1, token_count=1,
        source_block_ids=["b0001"], source_pages=[1],
    )
    payload = build_index_payload(
        document_id="doc123", title="Doc", source_path="/x/doc.pdf", markdown_path="", report_path="",
        source_hash="h", normalized_hash="h2", extractor="pdf", ocr_used=False, semantic_structuring_used=False,
        config_fingerprint="fp", embedding_info=FakeEmbeddingInfo(), chunks=[chunk], embeddings=[[0.1, 0.2]],
        retrieval_ready=True, warnings=0, errors=0,
    )
    assert payload["chunks"][0]["content_kind"] == CONTENT_KIND_MAIN_TEXT
    assert payload["visual"]["enabled"] is False


def test_build_index_payload_appends_visual_records_alongside_main_text(tmp_path):
    class FakeEmbeddingInfo:
        provider = "sentence-transformers-service"
        model = "BAAI/bge-large-en-v1.5"
        revision = "rev"
        dimension = 2
        max_sequence_tokens = 512
        normalized_embeddings = True
        query_prompt_name = None
        tokenizer_id = "BAAI/bge-large-en-v1.5"

    chunk = ChunkRecord(
        chunk_id="c1", chunk_index=0, header_path=["Intro"], heading="Intro",
        provenance="[doc.pdf > Intro]", text="hello", body="hello", word_count=1, token_count=1,
        source_block_ids=["b0001"], source_pages=[1],
    )
    visual_rec = build_visual_record(
        visual_id="v-doc123-abc", document_id="doc123", source_path="/x/doc.pdf", source_hash="h",
        page=1, bbox=[0, 0, 1, 1], visual_type="bar_chart", heading_path=["Intro"], linked_chunk_ids=["c1"],
        image_path="/x/_visuals/abc.png", image_hash="abc", analysis_model="qwen3-vl:8b-instruct-q4_K_M",
        analysis_model_digest=None, analysis_version="1", retrieval_text="Visual type: Bar chart.",
        analysis={}, document_order=100001, embedding=[0.3, 0.4],
    )
    payload = build_index_payload(
        document_id="doc123", title="Doc", source_path="/x/doc.pdf", markdown_path="", report_path="",
        source_hash="h", normalized_hash="h2", extractor="pdf", ocr_used=False, semantic_structuring_used=False,
        config_fingerprint="fp", embedding_info=FakeEmbeddingInfo(), chunks=[chunk], embeddings=[[0.1, 0.2]],
        retrieval_ready=True, warnings=0, errors=0,
        visual_records=[visual_rec], visual_stats={"enabled": True, "found": 1, "eligible": 1, "indexed": 1},
    )
    kinds = [c["content_kind"] for c in payload["chunks"]]
    assert kinds == [CONTENT_KIND_MAIN_TEXT, CONTENT_KIND_VISUAL_ANALYSIS]
    # MAIN_TEXT chunk text is untouched by the visual record's presence --
    # the MAIN_TEXT-regression invariant, enforced structurally here too.
    assert payload["chunks"][0]["text"] == "hello"


# ---------------------------------------------------------------------- #
# token-native visual-text sizing (2026-08-13 correction phase) --
# `build_retrieval_text_within_budget` must be the final correctness
# boundary, validated against the *active embedding backend's own*
# tokenizer/limit, never a hardcoded constant.

def _dense_diagram_analysis():
    variables = ["Day", "Month", "Hour", "Season", "Year", "Latitude", "Longitude",
                 "Altitude", "Region", "Zone", "Type", "CVD60", "tp", "ssr"]
    long_relationships = [
        f"{v} influences " + ", ".join(x for x in variables if x != v) + "."
        for v in variables
    ]
    raw = json.dumps({
        "visual_type": "diagram",
        "title": "Variable dependency network",
        "summary": "A dependency diagram showing how meteorological and temporal "
                    "variables influence predictions across the study region.",
        "axes": {"x": {"label": "", "units": ""}, "y": {"label": "", "units": ""}},
        "legend": ["Predictor", "Target"],
        "relationships": long_relationships,
        "key_takeaway": "All variables in the network are mutually influential.",
    })
    return visual_analysis.parse_and_validate(raw)


def test_build_retrieval_text_within_budget_noop_when_already_fits():
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    sizing = visual_analysis.build_retrieval_text_within_budget(
        analysis, ["Results"], 6, count_tokens=lambda t: len(t.split()), max_tokens=512,
    )
    assert sizing.fits_budget is True
    assert sizing.reduction_steps_applied == []
    assert sizing.token_count <= 512


def test_build_retrieval_text_within_budget_reduces_using_real_tokenizer_shape():
    # A strict fake tokenizer (character count, not word count) simulating
    # a tight budget forces real reduction steps to run.
    analysis = _dense_diagram_analysis()
    sizing = visual_analysis.build_retrieval_text_within_budget(
        analysis, ["Section"], 11, count_tokens=lambda t: len(t), max_tokens=400,
    )
    assert sizing.fits_budget is True
    assert sizing.reduction_steps_applied  # at least one reduction step ran
    assert sizing.token_count <= 400
    # Provenance is never dropped, even under heavy reduction.
    assert "Visual type:" in sizing.text
    assert "Page: 11." in sizing.text
    assert "Section: Section." in sizing.text


def test_build_retrieval_text_within_budget_never_reduces_below_provenance():
    # An impossibly small budget (even the header alone won't fit) --
    # every reduction step runs, provenance fields are still never
    # stripped, and the function reports fits_budget=False rather than
    # fabricating a shorter paraphrase or slicing mid-field.
    analysis = _dense_diagram_analysis()
    sizing = visual_analysis.build_retrieval_text_within_budget(
        analysis, ["Section"], 11, count_tokens=lambda t: len(t), max_tokens=5,
    )
    assert sizing.fits_budget is False
    assert set(sizing.reduction_steps_applied) == {name for name, _ in visual_analysis._TOKEN_BUDGET_REDUCTION_STEPS}
    assert "Visual type:" in sizing.text
    assert "Page: 11." in sizing.text


def test_build_retrieval_text_within_budget_handles_missing_tokenizer():
    # No embedding client resolvable (matches pipeline.py's fallback when
    # `client.health()` fails) -- returns the word-budget-only text as-is,
    # never crashes.
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    sizing = visual_analysis.build_retrieval_text_within_budget(
        analysis, ["Results"], 6, count_tokens=None, max_tokens=None,
    )
    assert sizing.fits_budget is True
    assert sizing.text == visual_analysis.build_retrieval_text(analysis, ["Results"], 6)


def test_build_retrieval_text_within_budget_reduction_order_drops_uncertainties_first():
    analysis = visual_analysis.parse_and_validate(json.dumps({
        "visual_type": "line_chart", "title": "T",
        "summary": "A short grounded summary of the chart.",
        "uncertainties": ["Some values were not fully legible."],
        "key_takeaway": "Overall usage increased over time.",
    }))
    full_text = visual_analysis.build_retrieval_text(analysis, ["Sec"], 3)
    full_tokens = len(full_text)
    # A budget just under the full text but comfortably above what's left
    # after dropping only "uncertainties" -- proves that field is reduced
    # before summary/key_takeaway/provenance.
    sizing = visual_analysis.build_retrieval_text_within_budget(
        analysis, ["Sec"], 3, count_tokens=lambda t: len(t), max_tokens=full_tokens - 5,
    )
    assert sizing.reduction_steps_applied[0] == "drop_uncertainties"
    assert "Key takeaway:" in sizing.text
    assert "Uncertain" not in sizing.text


# ---------------------------------------------------------------------- #
# cache isolation (2026-08-13 correction phase): the visual crop/analysis
# cache must derive from the *active* index_dir, never default globally to
# `docs/.index/_visuals` regardless of where the target index actually is.

def _fake_visual_analysis(monkeypatch):
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    monkeypatch.setattr(visual_analysis, "analyze_visual", lambda image_bytes, config: (analysis, "{}"))
    monkeypatch.setattr(visual_analysis, "get_model_digest", lambda config: None)
    monkeypatch.setattr(visual_analysis, "unload_model", lambda config: True)


def test_visual_cache_is_isolated_to_the_configured_index_dir(tmp_path, monkeypatch):
    from document_converter.config import PipelineConfig, REPO_ROOT
    from document_converter.pipeline import DocumentPipeline
    from document_converter.tests.conftest import FakeEmbeddingClient

    _fake_visual_analysis(monkeypatch)

    real_production_index = REPO_ROOT / "docs" / ".index"
    before = set(real_production_index.rglob("*")) if real_production_index.exists() else set()

    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    pdf_path = source_dir / "chart.pdf"
    _make_synthetic_pdf(str(pdf_path))

    validation_index_dir = tmp_path / "validation_index"
    other_index_dir = tmp_path / "other_validation_index"  # a second isolated target, never touched by the first

    config = PipelineConfig()
    config.paths.source_dir = str(source_dir)
    config.paths.processed_dir = str(tmp_path / "processed")
    config.paths.report_dir = str(tmp_path / "reports")
    config.paths.index_dir = str(validation_index_dir)
    config.paths.legacy_docs_dir = str(tmp_path / "legacy_docs")
    config.paths.ensure()
    config.semantic.mode = "off"
    config.visual.enabled = True
    # This test is scoped to proving cache-file isolation, not MAIN_TEXT
    # retrieval-readiness gating (a separate, already-covered concern) --
    # the synthetic image-only fixture has no body text to satisfy that
    # gate.
    config.strict = False

    pipeline = DocumentPipeline(config)
    pipeline._embedding_client = FakeEmbeddingClient()
    result = pipeline.index(str(pdf_path))

    assert result.visual_stats["indexed"] == 1
    assert (validation_index_dir / "_visuals").is_dir()
    assert any((validation_index_dir / "_visuals").iterdir())

    # The other isolated target was never created or touched by this run.
    assert not other_index_dir.exists()

    # The real repo's production index was not touched by this run.
    after = set(real_production_index.rglob("*")) if real_production_index.exists() else set()
    assert before == after
    assert not (real_production_index / "_visuals").exists()


# ---------------------------------------------------------------------- #
# Strict sequential GPU model lifecycle (RAG_BUILD_SOURCE_OF_TRUTH.md,
# "strict sequential GPU model lifecycle" phase): the VLM must be fully
# unloaded -- and, when nvidia-smi is available, a VRAM barrier evaluated --
# *before* any embedding call, including the visual-text embedding call
# that used to happen inside the old single `_process_visuals` before the
# VLM's own unload. These tests use the real `DocumentPipeline.index()`
# end-to-end (like the cache-isolation test above), recording the actual
# call order every stage-owning primitive is invoked in.

def _pipeline_config_for_lifecycle_test(tmp_path):
    from document_converter.config import PipelineConfig

    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    config = PipelineConfig()
    config.paths.source_dir = str(source_dir)
    config.paths.processed_dir = str(tmp_path / "processed")
    config.paths.report_dir = str(tmp_path / "reports")
    config.paths.index_dir = str(tmp_path / "index")
    config.paths.legacy_docs_dir = str(tmp_path / "legacy_docs")
    config.paths.ensure()
    config.semantic.mode = "off"
    config.visual.enabled = True
    config.strict = False
    return config, source_dir


def _make_synthetic_pdf_with_text_and_chart(path: str):
    """Like `_make_synthetic_pdf` above, but with a real body paragraph too
    -- these lifecycle-ordering tests need at least one actual MAIN_TEXT
    chunk to embed (the image-only fixture produces zero MAIN_TEXT chunks,
    which would make "embed happens before VLM unload" untestable)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Introduction", fontsize=16)
    page.insert_text(
        (72, 130),
        "This is a body paragraph with enough real words to form a genuine "
        "retrieval chunk of prose describing the topic in some detail.",
        fontsize=11,
    )
    chart_rect = pymupdf.Rect(100, 200, 500, 500)
    chart_pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300))
    chart_pix.set_rect(chart_pix.irect, (30, 100, 200))
    page.insert_image(chart_rect, pixmap=chart_pix)
    doc.save(path)
    doc.close()


class _OrderRecordingEmbeddingClient:
    """A FakeEmbeddingClient that appends to a shared `events` list on every
    call, so a test can assert relative ordering against other
    lifecycle-owning calls (VLM unload, VRAM barrier) recorded into the same
    list."""

    def __init__(self, events, dimension: int = 8, max_sequence_tokens: int = 512):
        self.events = events
        self.dimension = dimension
        self.max_sequence_tokens = max_sequence_tokens
        self.embed_calls = []
        self.release_calls = 0

    def health(self):
        from document_converter.embedding_client import EmbeddingModelInfo
        return EmbeddingModelInfo(
            provider="fake", model="fake-model", dimension=self.dimension,
            max_sequence_tokens=self.max_sequence_tokens, normalized_embeddings=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector_for(self, text: str) -> list:
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[: self.dimension]]

    def embed_batch(self, texts, reject_oversized: bool = True, input_type: str = "document"):
        self.embed_calls.append(list(texts))
        kind = "visual_embed" if input_type == "document" and texts and texts[0].startswith("Visual type:") else "embed"
        self.events.append(kind)
        return [self._vector_for(t) for t in texts]

    def embed_one(self, text: str, input_type: str = "document") -> list:
        return self.embed_batch([text], input_type=input_type)[0]

    def release(self) -> bool:
        self.release_calls += 1
        self.events.append("release")
        return True


def test_vlm_unload_happens_before_visual_text_embedding(tmp_path, monkeypatch):
    """The exact ordering bug this phase fixes: the old `_process_visuals`
    called `client.embed_batch(visual_texts, ...)` before
    `visual_analysis.unload_model()`, so the VLM and the embedding model
    could be GPU-resident together during that call. Proves the new,
    strict order: Stage 2 (VLM analyze -> unload -> VRAM barrier) runs to
    completion before Stage 3 does *any* embedding call -- MAIN_TEXT's or
    the visual text's -- and the embedding model is explicitly released at
    the end of Stage 3."""
    from document_converter.config import REPO_ROOT
    from document_converter.pipeline import DocumentPipeline

    config, source_dir = _pipeline_config_for_lifecycle_test(tmp_path)
    pdf_path = source_dir / "chart.pdf"
    _make_synthetic_pdf_with_text_and_chart(str(pdf_path))

    events: list = []
    analysis = visual_analysis.parse_and_validate(VALID_RESPONSE)
    monkeypatch.setattr(visual_analysis, "analyze_visual", lambda image_bytes, cfg: (analysis, "{}"))
    monkeypatch.setattr(visual_analysis, "get_model_digest", lambda cfg: None)

    def fake_unload(cfg):
        events.append("vlm_unload")
        return True

    monkeypatch.setattr(visual_analysis, "unload_model", fake_unload)
    # No real GPU/Ollama available in CI -- keep the barrier calls fast and
    # deterministic without disabling the ordering they participate in.
    monkeypatch.setattr(gpu_lifecycle, "read_gpu_memory_used_mib", lambda gpu_index=0: None)
    monkeypatch.setattr(
        gpu_lifecycle, "wait_for_ollama_model_unload",
        lambda host, model, timeout_s=30.0, poll_interval_s=1.0: events.append("vram_wait_ollama_ps") or True,
    )

    pipeline = DocumentPipeline(config)
    pipeline._embedding_client = _OrderRecordingEmbeddingClient(events)
    result = pipeline.index(str(pdf_path))

    assert result.visual_stats["indexed"] == 1
    # VLM unload (+ its ollama-ps barrier check) completes first, strictly
    # before any embedding call -- MAIN_TEXT's or the visual text's -- and
    # the embedding model is explicitly released once both are done.
    assert events == ["vlm_unload", "vram_wait_ollama_ps", "embed", "visual_embed", "release"]
    # And explicitly: never the old bug's order (visual-text embed while
    # the VLM might still be resident, i.e. before its unload call).
    assert events.index("vlm_unload") < events.index("visual_embed")


def test_vlm_unload_still_runs_when_analysis_raises_unexpectedly(tmp_path, monkeypatch):
    """Spec 'FAILURE HANDLING': an unexpected (non-VisualAnalysisError)
    failure mid-batch must still unload the VLM, not leave it resident."""
    from document_converter.config import PipelineConfig
    from document_converter.pipeline import DocumentPipeline

    config, source_dir = _pipeline_config_for_lifecycle_test(tmp_path)
    pdf_path = source_dir / "chart.pdf"
    _make_synthetic_pdf(str(pdf_path))

    unload_calls = []
    monkeypatch.setattr(visual_analysis, "get_model_digest", lambda cfg: None)
    monkeypatch.setattr(visual_analysis, "unload_model", lambda cfg: unload_calls.append(1) or True)
    monkeypatch.setattr(gpu_lifecycle, "read_gpu_memory_used_mib", lambda gpu_index=0: None)
    monkeypatch.setattr(gpu_lifecycle, "wait_for_ollama_model_unload", lambda *a, **k: True)

    def boom(image_bytes, cfg):
        raise RuntimeError("unexpected VLM crash, not a VisualAnalysisError")

    monkeypatch.setattr(visual_analysis, "analyze_visual", boom)

    pipeline = DocumentPipeline(config)
    pipeline._embedding_client = FakeEmbeddingClient()
    with pytest.raises(RuntimeError, match="unexpected VLM crash"):
        pipeline.index(str(pdf_path))

    assert unload_calls == [1]


def test_visual_embedding_failure_never_touches_already_written_main_text(tmp_path, monkeypatch):
    """Spec 'FAILURE BEHAVIOR': a Stage-3 visual-embedding failure degrades
    to fewer/no visual records, never corrupts or blocks the already-
    embedded MAIN_TEXT chunks."""
    from document_converter.config import PipelineConfig
    from document_converter.pipeline import DocumentPipeline
    from document_converter.embedding_client import EmbeddingServiceError

    config, source_dir = _pipeline_config_for_lifecycle_test(tmp_path)
    pdf_path = source_dir / "chart.pdf"
    _make_synthetic_pdf_with_text_and_chart(str(pdf_path))

    _fake_visual_analysis(monkeypatch)

    class FlakyOnVisualTextClient(FakeEmbeddingClient):
        def embed_batch(self, texts, reject_oversized: bool = True, input_type: str = "document"):
            if texts and texts[0].startswith("Visual type:"):
                raise EmbeddingServiceError("simulated visual-embedding outage")
            return super().embed_batch(texts, reject_oversized=reject_oversized, input_type=input_type)

    pipeline = DocumentPipeline(config)
    pipeline._embedding_client = FlakyOnVisualTextClient()
    result = pipeline.index(str(pdf_path))

    assert result.index_path and os.path.exists(result.index_path)
    payload = load_index(result.index_path)
    main_text_chunks = [c for c in payload["chunks"] if c["content_kind"] == CONTENT_KIND_MAIN_TEXT]
    assert main_text_chunks and all(c["embedding"] for c in main_text_chunks)
    assert payload["visual"]["indexed"] == 0
    assert payload["visual"]["failed"] >= 1


def test_embedding_client_released_at_end_of_ingestion(tmp_path, monkeypatch):
    """Spec 'EXPLICITLY RELEASE EMBEDDING MODEL': index() calls
    `client.release()` once Stage 3 (MAIN_TEXT + visual embedding) is
    done, for every document -- not only ones with visual content."""
    from document_converter.config import PipelineConfig
    from document_converter.pipeline import DocumentPipeline

    config, source_dir = _pipeline_config_for_lifecycle_test(tmp_path)
    config.visual.enabled = False
    md_path = source_dir / "doc.md"
    md_path.write_text("# Title\n\nSome content for indexing.\n", encoding="utf-8")

    pipeline = DocumentPipeline(config)
    fake = FakeEmbeddingClient()
    pipeline._embedding_client = fake
    result = pipeline.index(str(md_path))

    assert fake.release_calls == 1
    assert result.gpu_lifecycle_evidence is not None
    assert result.gpu_lifecycle_evidence["embedding_stage_released"] is True
