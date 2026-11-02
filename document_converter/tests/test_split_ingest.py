"""
Tests for the pre-ingest PDF splitter (document_converter/split_ingest.py) --
see the module's own docstring for the split algorithm and the H1-source
(Option A / Option B) rules being exercised here.

Two fixtures are used, deliberately for different reasons:

* `fixture_pdf` is the *literal* fixture given in the task spec (8 sections,
  60 pages each, a real level-1 outline). It's used for every test that only
  cares about the splitter's own arithmetic -- page coverage, section
  boundaries, byte sizes, determinism -- none of which depend on what the
  extractor does with the page *content*.
* `_retrieval_fixture` is a separate, content-richer fixture used only for
  the end-to-end ingest+search test. The spec fixture's per-page repeated
  "Section N" heading and "Body line {i} section {sec}." text is, once PDF
  extraction's digit-normalized repeated-header/footer detector runs
  (`pdf_extractor.py`'s `_detect_repeated_running_lines` -- an existing,
  unrelated heuristic this task does not touch), indistinguishable from a
  templated page header/footer and gets stripped almost entirely -- it was
  never meant to survive full-pipeline extraction, only to exercise the
  splitter's own page/byte arithmetic. The retrieval fixture uses distinct,
  non-numeric wording per section/page instead so real content survives
  extraction, which is what that one test needs to prove.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from control import BM25, clean_query
from document_converter.config import PipelineConfig
from document_converter.index_schema import load_index
from document_converter.split_ingest import (
    TARGET_PART_BYTES,
    SplitError,
    Unit,
    _build_children,
    resolve_h1_boundaries,
    split_and_ingest,
    split_pdf,
)

SECTIONS = 8
PAGES_PER_SECTION = 60

# The spec fixture's ~5MB target would collapse this whole (garbage-
# collected) ~170KB, text-only fixture into a single part and never
# exercise the accumulate-and-close logic -- these tests pass a
# proportionally scaled-down `target_bytes` instead. `TARGET_PART_BYTES`
# itself (the real ~5MB default) is exercised directly in
# `test_default_target_is_five_megabytes`.
SMALL_TARGET = 50_000
TINY_TARGET = 10_000  # smaller than any single 60-page section (~20.5KB)


def _build_spec_fixture(path: Path, sections: int = SECTIONS, pages_per_section: int = PAGES_PER_SECTION) -> None:
    """Exactly the snippet given in the task spec."""
    d = fitz.open()
    toc = []
    page = 0
    for sec in range(sections):
        toc.append([1, f"Section {sec + 1}", page + 1])
        for p in range(pages_per_section):
            pg = d.new_page()
            pg.insert_text((72, 60), f"Section {sec + 1}", fontsize=18)
            for i in range(20):
                pg.insert_text((72, 95 + i * 16), f"Body line {i} section {sec + 1}.", fontsize=11)
            page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()


@pytest.fixture
def fixture_pdf(tmp_path) -> Path:
    path = tmp_path / "split_smoke.pdf"
    _build_spec_fixture(path)
    return path


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


# --------------------------------------------------------------------------
# Step 1: H1 source (Option A / Option B)

def test_outline_h1_boundaries_resolved_directly_option_a(fixture_pdf, config):
    boundaries, source = resolve_h1_boundaries(fixture_pdf, config)
    assert source == "outline"
    assert [title for _, title in boundaries] == [f"Section {i + 1}" for i in range(SECTIONS)]
    assert [page for page, _ in boundaries] == [i * PAGES_PER_SECTION for i in range(SECTIONS)]


def test_folder_only_outline_entries_resolve_via_first_resolvable_descendant(tmp_path, config):
    """Real-world outlines (confirmed against sql-sql-server-ver17.pdf: 20 of
    its 23 level-1 entries) commonly nest level-1 "folder" bookmarks that
    carry no destination page of their own (`kind: 0, page: -1` in
    `get_toc(simple=False)`), grouping only deeper real destinations
    beneath them. Still Option A -- resolved from the outline tree alone,
    no extraction -- never a fall-through to Option B."""
    path = tmp_path / "folder_outline.pdf"
    d = fitz.open()
    for _ in range(6):
        d.new_page()
    # Level-1 "Folder" has no destination of its own; its two children (a
    # real page-2 destination and a level-1-shaped-but-nested entry) sit
    # between it and the next level-1 entry "Chapter B".
    toc = [
        [1, "Folder", -1],
        [2, "Folder child", 2],
        [1, "Chapter B", 5],
    ]
    d.set_toc(toc)
    d.save(str(path))
    d.close()

    boundaries, source = resolve_h1_boundaries(path, config)
    assert source == "outline"
    assert boundaries == [(1, "Folder"), (4, "Chapter B")]


def test_option_b_fallback_finds_the_extractors_single_title_level_heading(tmp_path, config):
    """Option B ("read level-1 HeadingBlock positions") only ever surfaces
    ONE boundary in this extractor: `HeadingBlock.level == 1` means the
    single resolved document *title* (`is_resolved_title` /
    `_resolve_heading_level` in pdf_extractor.py), never a repeatable
    chapter/section tier -- those land at level 2+. A PDF with no outline
    therefore falls back to exactly one section (the whole document) even
    when it visibly has several same-styled "chapter" headings -- see the
    split_ingest module docstring and the written report for why this
    makes Option A's outline-tree reading the load-bearing path, not just
    the preferred one."""
    path = tmp_path / "no_outline.pdf"
    d = fitz.open()
    topics = ["Frobnication overview", "Widgetry overview", "Kerning overview"]
    for topic in topics:
        for p in range(4):
            pg = d.new_page()
            if p == 0:
                pg.insert_text((72, 60), topic, fontsize=18)
            for i in range(6):
                pg.insert_text((72, 100 + i * 16), f"{topic} body detail line about topic depth {i}.", fontsize=11)
    d.save(str(path))  # no set_toc() -- no outline at all
    d.close()

    boundaries, source = resolve_h1_boundaries(path, config)
    assert source == "extraction"
    assert len(boundaries) == 1
    assert boundaries[0][0] == 0


def test_no_h1_anywhere_falls_back_to_a_single_section(tmp_path, config):
    path = tmp_path / "flat.pdf"
    d = fitz.open()
    for pno in range(3):
        pg = d.new_page()
        for i in range(8):
            pg.insert_text((72, 72 + i * 16),
                            f"this is ordinary paragraph body text, line {i} of page {pno}.", fontsize=11)
    d.save(str(path))
    d.close()

    parts, source = split_pdf(path, tmp_path / "out", config)
    assert source == "none"
    assert len(parts) == 1
    assert parts[0].page_count == 3


# --------------------------------------------------------------------------
# Split algorithm: coverage, section integrity, size behavior, determinism

def test_coverage_invariant_no_pages_dropped_or_duplicated(fixture_pdf, tmp_path, config):
    parts, _ = split_pdf(fixture_pdf, tmp_path / "out", config, target_bytes=SMALL_TARGET)

    src = fitz.open(str(fixture_pdf))
    total_source_pages = src.page_count
    src.close()

    seen = set()
    total_part_pages = 0
    for part in parts:
        page_range = set(range(part.start_page, part.end_page + 1))
        assert not (page_range & seen), "a page appears in two parts"
        seen |= page_range
        total_part_pages += part.page_count
        on_disk = fitz.open(part.path)
        assert on_disk.page_count == part.page_count
        on_disk.close()

    assert total_part_pages == total_source_pages
    assert seen == set(range(total_source_pages))


def test_section_integrity_every_part_starts_at_an_h1_or_page_one(fixture_pdf, tmp_path, config):
    boundaries, _ = resolve_h1_boundaries(fixture_pdf, config)
    boundary_pages = {page for page, _ in boundaries}
    parts, _ = split_pdf(fixture_pdf, tmp_path / "out", config, target_bytes=SMALL_TARGET)
    assert len(parts) > 1, "fixture should produce multiple parts at this target"
    for part in parts:
        assert part.start_page == 0 or part.start_page in boundary_pages


def test_sections_are_never_cut_across_parts(fixture_pdf, tmp_path, config):
    parts, _ = split_pdf(fixture_pdf, tmp_path / "out", config, target_bytes=SMALL_TARGET)
    for part in parts:
        assert part.start_page % PAGES_PER_SECTION == 0
        assert (part.end_page + 1) % PAGES_PER_SECTION == 0


def test_parts_close_before_exceeding_target_when_sections_allow_it(fixture_pdf, tmp_path, config):
    parts, _ = split_pdf(fixture_pdf, tmp_path / "out", config, target_bytes=SMALL_TARGET)
    assert len(parts) > 1
    print(f"\n{len(parts)} part(s) at target={SMALL_TARGET}B:")
    for part in parts:
        print(f"  part {part.part_index}: pages {part.start_page}-{part.end_page} "
              f"({part.page_count}p) {part.size_bytes} bytes -- {part.section_titles}")
        # Every part here is 2 whole sections; growth stopped only because a
        # THIRD section would have pushed it over -- never because the part
        # itself already exceeded the target.
        assert part.size_bytes <= SMALL_TARGET


def test_oversized_leaf_with_no_children_fails_loudly_with_independently_verified_span(fixture_pdf, tmp_path, config):
    """Supersedes the pre-recursion design's "a single oversized section is
    allowed to exceed the target" behavior: the spec fixture's 8 sections are
    a genuinely flat, single-level outline (no nested entries under any
    "Section N"), so at TINY_TARGET every one of them is both oversized AND
    childless -- exactly the CRITICAL FLOOR BEHAVIOR case. The splitter must
    now fail loudly instead of silently emitting an oversized part.

    The expected page range is computed here from PAGES_PER_SECTION alone --
    never by calling `resolve_h1_boundaries`/`split_pdf`'s own resolution
    code -- so this can't pass by reproducing the same bug on both sides."""
    with pytest.raises(SplitError) as excinfo:
        split_pdf(fixture_pdf, tmp_path / "out", config, target_bytes=TINY_TARGET)

    err = excinfo.value
    assert err.pdf_path == str(fixture_pdf)
    assert err.title == "Section 1"
    assert err.path == ["Section 1"]
    assert err.level == 1
    # Independently derived: section 0 spans pages [0, PAGES_PER_SECTION) 0-indexed.
    assert err.first_page == 1
    assert err.last_page == PAGES_PER_SECTION
    assert err.size_bytes > TINY_TARGET
    assert err.target_bytes == TINY_TARGET
    assert "no outline children" in err.reason


def test_default_target_is_five_megabytes():
    assert TARGET_PART_BYTES == 5 * 1024 * 1024


def test_determinism_same_boundaries_and_sizes_across_runs(fixture_pdf, tmp_path, config):
    parts1, source1 = split_pdf(fixture_pdf, tmp_path / "out1", config, target_bytes=SMALL_TARGET)
    parts2, source2 = split_pdf(fixture_pdf, tmp_path / "out2", config, target_bytes=SMALL_TARGET)
    assert source1 == source2
    assert [(p.start_page, p.end_page, p.size_bytes) for p in parts1] == \
           [(p.start_page, p.end_page, p.size_bytes) for p in parts2]


def test_front_matter_before_first_h1_goes_into_part_one(tmp_path, config):
    path = tmp_path / "frontmatter.pdf"
    d = fitz.open()
    for _ in range(3):  # cover + TOC pages, no heading
        pg = d.new_page()
        pg.insert_text((72, 72), "Cover page text.", fontsize=12)
    toc = []
    page = 3
    for sec in range(2):
        toc.append([1, f"Chapter {sec + 1}", page + 1])
        for p in range(5):
            pg = d.new_page()
            pg.insert_text((72, 72), f"Chapter {sec + 1} body {p}.", fontsize=12)
            page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()

    parts, source = split_pdf(path, tmp_path / "out", config)
    assert source == "outline"
    assert parts[0].start_page == 0  # the 3 front-matter pages are in part 1
    assert parts[0].section_titles[0] == "Front matter"


def test_no_hardcoded_page_or_part_count_limit(tmp_path, config):
    """A document with far more top-level sections than the smoke fixture
    still just produces more parts -- nothing here caps section/page/part
    count."""
    path = tmp_path / "many_sections.pdf"
    d = fitz.open()
    toc = []
    for sec in range(40):
        toc.append([1, f"Section {sec + 1}", sec + 1])
        pg = d.new_page()
        pg.insert_text((72, 72), f"Section {sec + 1} content.", fontsize=11)
    d.set_toc(toc)
    d.save(str(path))
    d.close()

    parts, source = split_pdf(path, tmp_path / "out", config, target_bytes=TARGET_PART_BYTES)
    assert source == "outline"
    total_pages = sum(p.page_count for p in parts)
    assert total_pages == 40


# --------------------------------------------------------------------------
# Recursive descent: real-world outline shapes (page-less parent folders,
# nested multiple levels deep) that require descending BELOW level 1 -- this
# is what makes sql-sql-server-ver17.pdf's ~76MB `Reference` monolith
# splittable at all (20 of its 23 level-1 entries are page-less folders, and
# `Reference` itself is one, containing 13 level-2 children nested as deep
# as level 6).

def _build_pageless_parent_fixture(path: Path, children: int = 4, pages_per_child: int = 8,
                                    appendix_pages: int = 2) -> int:
    """A single-level page-less-folder fixture: level-1 "Reference" carries
    NO destination of its own (`page == -1`, exactly like the real file's
    folder-only entries), its 4 level-2 children each carry a real
    destination page, and a level-1 "Appendix" (real page) follows. Returns
    the total page count."""
    d = fitz.open()
    page = 0
    toc = [[1, "Reference", -1]]
    for c in range(children):
        title = f"RefChild {c + 1}"
        toc.append([2, title, page + 1])
        for _ in range(pages_per_child):
            pg = d.new_page()
            pg.insert_text((72, 60), title, fontsize=18)
            for i in range(20):
                pg.insert_text((72, 95 + i * 16), f"{title} body line {i}.", fontsize=11)
            page += 1
    toc.append([1, "Appendix", page + 1])
    for _ in range(appendix_pages):
        pg = d.new_page()
        pg.insert_text((72, 60), "Appendix", fontsize=18)
        for i in range(20):
            pg.insert_text((72, 95 + i * 16), f"Appendix body line {i}.", fontsize=11)
        page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()
    return page


# Empirically calibrated (see task notes) against `_build_pageless_parent_fixture`'s
# default shape: small enough that "Reference" measured whole (32 pages) must
# recurse, large enough that two 8-page children packed together still fit.
PAGELESS_RECURSION_TARGET = 13_000


def test_pageless_folder_toc_entry_reproduces_real_kind0_page_minus1_shape(tmp_path):
    """Confirms `d.set_toc()` with `page=-1` actually produces the same raw
    outline shape pymupdf reports for the real file's folder-only entries
    (`kind: 0, page: -1`) -- i.e. this fixture is not a normal bookmark in
    disguise."""
    path = tmp_path / "shape_check.pdf"
    _build_pageless_parent_fixture(path, children=1, pages_per_child=1, appendix_pages=1)
    pdf = fitz.open(str(path))
    toc = pdf.get_toc(simple=False)
    pdf.close()
    reference_entry = next(e for e in toc if e[1] == "Reference")
    assert reference_entry[0] == 1
    assert reference_entry[2] == -1
    assert reference_entry[3]["kind"] == 0  # 0 == no destination; a real GoTo dest is kind 4


def test_pageless_oversized_parent_recursively_descends_into_its_children(tmp_path, config):
    """The intersection this task specifically requires: a page-less top-
    level folder ("Reference") whose OWN resolved span (via descendant
    resolution) is too large for the target, is NOT accepted as one giant
    part -- its real level-2 children provide the boundaries, and those
    children get packed into multiple output parts."""
    path = tmp_path / "pageless_recursion.pdf"
    total_pages = _build_pageless_parent_fixture(path)

    boundaries, source = resolve_h1_boundaries(path, config)
    assert source == "outline"
    # "Reference" resolves (via its first child's page) to page 0, even
    # though its own outline entry carries no destination at all.
    assert boundaries == [(0, "Reference"), (32, "Appendix")]

    parts, source = split_pdf(path, tmp_path / "out", config, target_bytes=PAGELESS_RECURSION_TARGET)
    assert source == "outline"

    # Recursion happened: no part is titled "Reference" itself (it was never
    # accepted whole), and more than one part came out of its children.
    all_titles = [t for p in parts for t in p.section_titles]
    assert "Reference" not in all_titles
    child_parts = [p for p in parts if any(t.startswith("RefChild") for t in p.section_titles)]
    assert len(child_parts) > 1, "Reference's children should have needed more than one part"
    for p in child_parts:
        assert p.boundary_level == 2  # the recursion landed one level below Reference's own level-1

    # Every one of the 4 children is present in some part, in order, with no
    # unit split apart from its own page range.
    covered_titles = [t for p in child_parts for t in p.section_titles]
    assert covered_titles == [f"RefChild {i + 1}" for i in range(4)]

    # Page coverage across the whole document -- including across the
    # recursion boundary -- is exact: no drop, no duplicate.
    seen = set()
    for p in parts:
        rng = set(range(p.start_page, p.end_page + 1))
        assert not (rng & seen), "a page appears in two parts"
        seen |= rng
    assert seen == set(range(total_pages))
    assert sum(p.page_count for p in parts) == total_pages

    # Part ordering matches document order (Appendix, the last top-level
    # unit, comes out last).
    assert parts[-1].section_titles == ["Appendix"]


def test_pageless_oversized_parent_recursion_is_deterministic(tmp_path, config):
    path = tmp_path / "pageless_recursion.pdf"
    _build_pageless_parent_fixture(path)
    parts1, source1 = split_pdf(path, tmp_path / "out1", config, target_bytes=PAGELESS_RECURSION_TARGET)
    parts2, source2 = split_pdf(path, tmp_path / "out2", config, target_bytes=PAGELESS_RECURSION_TARGET)
    assert source1 == source2
    assert [(p.start_page, p.end_page, p.size_bytes, tuple(p.section_titles)) for p in parts1] == \
           [(p.start_page, p.end_page, p.size_bytes, tuple(p.section_titles)) for p in parts2]


def _build_multilevel_fixture(path: Path, leaves: int = 4, pages_per_leaf: int = 8,
                               appendix_pages: int = 2) -> int:
    """Three real outline levels: level-1 "Reference" (page-less) contains a
    single level-2 "BigPart" (also page-less) which contains `leaves`
    level-3 children (real pages). "Reference" and "BigPart" resolve to the
    exact same span (BigPart is their only child), so a oversized-parent
    check at level 1 immediately re-triggers at level 2 before finally
    bottoming out at level-3 units small enough to pack -- exercising
    descent through MORE than one level, with no depth limit assumed."""
    d = fitz.open()
    page = 0
    toc = [[1, "Reference", -1], [2, "BigPart", -1]]
    for c in range(leaves):
        title = f"Leaf {c + 1}"
        toc.append([3, title, page + 1])
        for _ in range(pages_per_leaf):
            pg = d.new_page()
            pg.insert_text((72, 60), title, fontsize=18)
            for i in range(20):
                pg.insert_text((72, 95 + i * 16), f"{title} body line {i}.", fontsize=11)
            page += 1
    toc.append([1, "Appendix", page + 1])
    for _ in range(appendix_pages):
        pg = d.new_page()
        pg.insert_text((72, 60), "Appendix", fontsize=18)
        for i in range(20):
            pg.insert_text((72, 95 + i * 16), f"Appendix body line {i}.", fontsize=11)
        page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()
    return page


MULTILEVEL_RECURSION_TARGET = 13_000


def test_recursive_descent_through_more_than_one_outline_level(tmp_path, config):
    """level 1 ("Reference") is oversized -> its only child, level 2
    ("BigPart"), is *also* oversized (same span, so recursing into it changes
    nothing about the size problem by itself) -> level 3 ("Leaf N") finally
    provides units small enough to pack. No depth limit is hard-coded
    anywhere in the splitter; this just has to actually bottom out."""
    path = tmp_path / "multilevel.pdf"
    total_pages = _build_multilevel_fixture(path)

    parts, source = split_pdf(path, tmp_path / "out", config, target_bytes=MULTILEVEL_RECURSION_TARGET)
    assert source == "outline"

    all_titles = [t for p in parts for t in p.section_titles]
    assert "Reference" not in all_titles
    assert "BigPart" not in all_titles  # neither intermediate level was ever accepted whole

    leaf_parts = [p for p in parts if any(t.startswith("Leaf") for t in p.section_titles)]
    assert len(leaf_parts) > 1
    for p in leaf_parts:
        assert p.boundary_level == 3  # bottomed out two levels below "Reference"

    covered_titles = [t for p in leaf_parts for t in p.section_titles]
    assert covered_titles == [f"Leaf {i + 1}" for i in range(4)]

    seen = set()
    for p in parts:
        rng = set(range(p.start_page, p.end_page + 1))
        assert not (rng & seen)
        seen |= rng
    assert seen == set(range(total_pages))
    assert sum(p.page_count for p in parts) == total_pages


def test_oversized_leaf_under_pageless_ancestor_fails_with_independently_derived_span(tmp_path, config):
    """Directly targets the failure mode the spec calls out by name: "a
    page-less ancestor or adjacent sibling causes the splitter to calculate
    the wrong end page." Both the unit that fails AND its neighbors are
    page-less-folder-wrapped, so a bug in end-page resolution through a
    page-less ancestor would corrupt the reported span in exactly the way
    this test would catch.

    The expected span is derived here purely from the fixture's own
    authored page numbers (K, EPILOGUE_START) -- never by calling
    `resolve_h1_boundaries`/`_build_children`/`split_pdf` -- so this can't
    pass by reproducing the same bug on both sides of the assertion."""
    K = 100  # BigLeaf's page count -- oversized at the chosen target, no children to recurse into
    path = tmp_path / "bigleaf_under_pageless.pdf"
    d = fitz.open()
    page = 0
    toc = [[1, "Prelude", -1], [2, "PreludeBody", 1]]
    pg = d.new_page()
    pg.insert_text((72, 60), "PreludeBody", fontsize=18)
    for i in range(20):
        pg.insert_text((72, 95 + i * 16), f"Prelude body line {i}.", fontsize=11)
    page += 1

    toc.append([1, "BigFolder", -1])
    toc.append([2, "BigLeaf", page + 1])
    for _ in range(K):
        pg = d.new_page()
        pg.insert_text((72, 60), "BigLeaf", fontsize=18)
        for i in range(20):
            pg.insert_text((72, 95 + i * 16), f"BigLeaf body line {i}.", fontsize=11)
        page += 1

    epilogue_start_1idx = page + 1  # independently known, authored constant
    toc.append([1, "Epilogue", epilogue_start_1idx])
    for _ in range(3):
        pg = d.new_page()
        pg.insert_text((72, 60), "Epilogue", fontsize=18)
        for i in range(20):
            pg.insert_text((72, 95 + i * 16), f"Epilogue body line {i}.", fontsize=11)
        page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()

    # Independently computed expected span -- pure arithmetic on the
    # fixture's own authored constants, not on any splitter code path.
    expected_first_page = 2  # BigLeaf's authored destination (1-indexed)
    expected_last_page = epilogue_start_1idx - 1  # everything up to the next sibling's start

    with pytest.raises(SplitError) as excinfo:
        split_pdf(path, tmp_path / "out", config, target_bytes=20_000)

    err = excinfo.value
    assert err.path == ["BigFolder", "BigLeaf"]
    assert err.level == 2
    assert err.first_page == expected_first_page
    assert err.last_page == expected_last_page
    assert err.size_bytes > 20_000
    assert "no outline children" in err.reason
    assert err.pdf_path == str(path)


# --------------------------------------------------------------------------
# The stated rule for content between adjacent child destinations: a child
# begins at its resolved destination page and its span ends immediately
# before the NEXT sibling's resolved destination -- pages in between belong
# to the earlier child even though nothing marks them individually. Checked
# directly against `_build_children` (pure outline-tree arithmetic, no PDF
# I/O) so this is not entangled with byte-size packing behavior at all.

def test_pages_between_adjacent_child_destinations_belong_to_the_earlier_child():
    # 10 pages (0-indexed 0-9). "Chapter A" destination is page 2 (1-indexed);
    # "Chapter B" destination is page 7 -- pages 2-6 (1-indexed) have no
    # marker of their own and must all be attributed to "Chapter A".
    toc = [[1, "Chapter A", 2], [1, "Chapter B", 7]]
    children = _build_children(toc, 0, len(toc), 0, 9, [])

    assert [(c.title, c.start_page, c.end_page, c.is_synthetic) for c in children] == [
        ("Front matter", 0, 0, True),     # page 0 (1-indexed page 1): before any destination
        ("Chapter A", 1, 5, False),        # pages 1-5 (0-indexed) == 1-indexed pages 2-6
        ("Chapter B", 6, 9, False),         # pages 6-9 (0-indexed) == 1-indexed pages 7-10
    ]


def test_pages_between_destinations_end_to_end_through_split_pdf(tmp_path, config):
    """Same rule, exercised through the real save/measure path: Chapter A's
    5-page span (only page 2 is individually "marked") is preserved intact
    as one part when it fits, not truncated to just its first authored
    page."""
    path = tmp_path / "gap.pdf"
    d = fitz.open()
    for _ in range(1):  # front matter page
        pg = d.new_page()
        pg.insert_text((72, 72), "Cover.", fontsize=12)
    toc = [[1, "Chapter A", 2], [1, "Chapter B", 7]]
    for p in range(5):  # Chapter A: pages 2-6 (1-indexed) -- only page 2 "authored" as the heading
        pg = d.new_page()
        if p == 0:
            pg.insert_text((72, 60), "Chapter A", fontsize=18)
        pg.insert_text((72, 100), f"Chapter A filler page {p}.", fontsize=11)
    for p in range(4):  # Chapter B: pages 7-10 (1-indexed)
        pg = d.new_page()
        if p == 0:
            pg.insert_text((72, 60), "Chapter B", fontsize=18)
        pg.insert_text((72, 100), f"Chapter B filler page {p}.", fontsize=11)
    d.set_toc(toc)
    d.save(str(path))
    d.close()

    parts, source = split_pdf(path, tmp_path / "out", config)  # default 5MB target: everything fits in 1 part
    assert source == "outline"
    assert len(parts) == 1
    assert parts[0].start_page == 0
    assert parts[0].end_page == 9
    assert parts[0].page_count == 10


# --------------------------------------------------------------------------
# Merge: ingest every part through DocumentPipeline.index(), confirm one
# shared, BM25-retrievable knowledge base.

_TOPICS = ["frobnication", "widgetry", "kerning", "compaction", "hashing", "caching", "indexing", "streaming"]
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
_LINEWORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
]


def _build_retrieval_fixture(path: Path, sections: int = 8, pages_per_section: int = 10) -> None:
    d = fitz.open()
    toc = []
    page = 0
    for sec in range(sections):
        topic = _TOPICS[sec]
        toc.append([1, f"{topic.capitalize()} overview", page + 1])
        for p in range(pages_per_section):
            pg = d.new_page()
            if p == 0:
                pg.insert_text((72, 60), f"{topic.capitalize()} overview", fontsize=18)
            ordinal = _ORDINALS[p]
            for i in range(20):
                word = _LINEWORDS[i]
                pg.insert_text((72, 95 + i * 16),
                                f"The {topic} subsystem on the {ordinal} page discusses {word} matters in detail.",
                                fontsize=11)
            page += 1
    d.set_toc(toc)
    d.save(str(path))
    d.close()


def test_split_and_ingest_merges_all_parts_into_one_index_retrievable_by_bm25(pipeline, tmp_path):
    retrieval_fixture = tmp_path / "retrieval_fixture.pdf"
    _build_retrieval_fixture(retrieval_fixture)

    results = split_and_ingest(
        pipeline, str(retrieval_fixture), work_dir=str(tmp_path / "parts"), target_bytes=200_000,
    )
    assert len(results) > 1, "fixture should split into multiple parts at this target"
    assert all(r["status"] == "indexed" for r in results), results

    # Every part is its own distinct source file -> its own document_id --
    # no dedup, no collision between parts (stable_document_id is keyed off
    # the part's own filename, per pipeline.py).
    doc_ids = [r["document_id"] for r in results]
    assert len(set(doc_ids)) == len(doc_ids)

    index_files = sorted(Path(pipeline.config.paths.index_dir).glob("*.json"))
    assert len(index_files) == len(results)

    chunk_texts = []
    for f in index_files:
        payload = load_index(str(f))
        chunk_texts.extend(c["text"] for c in payload["chunks"])

    # Chunks from every one of the 8 sections made it into the shared index,
    # not just the parts that happened to be ingested first/last.
    for topic in _TOPICS:
        assert any(topic in t.lower() for t in chunk_texts), f"no chunk found for topic {topic!r}"

    # Criterion 5: a query hitting section 8's ("streaming") body returns a
    # chunk from that section, via the existing BM25 ranker (control.py).
    corpus = [clean_query(t).split() for t in chunk_texts]
    bm25 = BM25(corpus)
    query_tokens = clean_query("streaming subsystem discusses matters").split()
    scores = bm25.get_scores(query_tokens)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert scores[best_idx] > 0
    assert "streaming" in chunk_texts[best_idx].lower()
