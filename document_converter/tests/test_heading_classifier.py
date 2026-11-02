"""
Unit tests for the shared heading-candidate classifier (extractors/__init__.py)
and its callers -- the code path responsible for the two confirmed regression
failures: arbitrary OCR fragments/code/equations/page-furniture being promoted
to headings, and legitimate numbered headings being flattened to one level.
"""
from __future__ import annotations

from document_converter.extractors import (
    heading_rejection_reason, is_heading_candidate, numbering_depth, numbering_tuple,
    numbering_prefix_remainder,
    looks_like_title_candidate, looks_like_algorithm_line, looks_like_code_or_algorithm_line,
    looks_like_equation_or_table_row, clean_title, looks_like_caption_text,
    looks_like_axis_or_legend_label, is_plausible_outline_successor, split_compound_numbered_heading,
    looks_like_figure_or_diagram_label, looks_like_diagram_annotation, looks_like_bare_quantity_label,
    looks_like_short_introductory_phrase, looks_like_table_cell_fragment, heading_capitalization_shape,
    ocr_heading_negative_signals, classify_ocr_nonheading_kind,
)


# ---------------------------------------------------------------------- #
# Rejection: code / algorithm syntax

def test_lone_brace_rejected_as_heading():
    assert not is_heading_candidate("}")


def test_function_signature_rejected_as_heading():
    assert not is_heading_candidate("LinearSearch(numbers, numbersSize, key) {")


def test_code_assignment_rejected_as_heading():
    assert heading_rejection_reason("keyIndex = 0") is not None
    assert heading_rejection_reason("i=0") is not None


def test_return_statement_rejected_as_heading():
    assert heading_rejection_reason("return -1 // not found") is not None


def test_print_call_rejected_as_heading():
    assert heading_rejection_reason('print ("NUMBERS: ")') is not None


def test_algorithm_line_number_rejected_as_heading():
    assert heading_rejection_reason("12: end function") == "code_or_algorithm"
    assert heading_rejection_reason("1: function s = LAPREORDERTREETRAVERSAL(S, /)") == "code_or_algorithm"


def test_end_function_rejected_regardless_of_numbering():
    assert heading_rejection_reason("end function") is not None


# ---------------------------------------------------------------------- #
# Rejection: equations / matrices / page numbers

def test_equation_rejected_as_heading():
    assert heading_rejection_reason('Your = Ag" ei, (1)') == "equation_or_matrix"


def test_equation_number_attached_and_rejected_as_heading():
    assert heading_rejection_reason("(3)") is not None
    assert heading_rejection_reason("(17)") is not None


def test_number_only_page_label_rejected_as_heading():
    assert heading_rejection_reason("94") == "no_letters"
    assert heading_rejection_reason("0") == "no_letters"
    assert heading_rejection_reason("2400") == "no_letters"


# ---------------------------------------------------------------------- #
# Rejection: OCR fragments / punctuation

def test_single_letter_fragment_rejected():
    assert heading_rejection_reason("t") is not None
    assert heading_rejection_reason("A") is not None


def test_two_letter_garbage_rejected():
    assert heading_rejection_reason("SU") == "ocr_fragment"
    assert heading_rejection_reason("Zu") == "ocr_fragment"


def test_isolated_punctuation_rejected():
    assert heading_rejection_reason("~") is not None
    assert heading_rejection_reason(">") is not None
    assert heading_rejection_reason("< >") is not None


# ---------------------------------------------------------------------- #
# Rejection: prose continuations (naive "short isolated line" false positives)

def test_prose_lead_in_sentence_rejected():
    assert heading_rejection_reason("As an example, the following vector s:") is not None
    assert heading_rejection_reason("is equivalent to the following operation on Ag:") is not None
    assert heading_rejection_reason("to v vertices in sy to be pushed to sp:") is not None


def test_caption_debris_with_internal_period_rejected():
    assert heading_rejection_reason("2.1.3: Linear search runtime. L") is not None
    assert heading_rejection_reason("GIVI 2.2.5: Speed of linear search versus binary search. LJ") is not None


# ---------------------------------------------------------------------- #
# Acceptance: legitimate headings must remain possible

def test_legitimate_short_headings_accepted():
    for text in ("Scope", "Risks", "Summary", "Abstract", "References", "Acknowledgement"):
        assert is_heading_candidate(text), f"{text!r} should be a valid heading candidate"


def test_numbered_section_headings_accepted():
    for text in (
        "1 Introduction", "2 Graphs in Linear Algebra", "3.1 Status of Vertices",
        "3.2 Tracking Vertex Visit Order", "6.2 Linear Algebraic Formulation",
        "2.11 LAB: Binary search with custom comparer",
    ):
        assert is_heading_candidate(text), f"{text!r} should be a valid heading candidate"


def test_real_title_accepted():
    assert is_heading_candidate("Linear Algebraic Depth-First Search")


# ---------------------------------------------------------------------- #
# numbering_depth / numbering_tuple

def test_numbering_depth_levels():
    assert numbering_depth("1 Introduction") == 1
    assert numbering_depth("3.1 Status of Vertices") == 2
    assert numbering_depth("3.1.1 Something") == 3
    assert numbering_depth("Abstract") is None
    assert numbering_depth("Chapter 3") == 1


def test_numbering_tuple_parses_outline_prefix():
    assert numbering_tuple("3.1.2 Foo") == (3, 1, 2)
    assert numbering_tuple("Abstract") is None


def test_dash_separated_number_is_enumerated_value_item_not_outline_numbering():
    """Exact strings from the anthology baseline's
    repeated_top_level_section_numbering findings: "N - text" lines are
    enumerated parameter values (option/value lists inside procedure steps
    and syntax references), not numbered section headings -- they must
    neither parse as outline numbering nor be promoted to headings."""
    for text in (
        "1 - Rowcount check only (the default)",
        "2 - Create a SQL Server credential using a shared",
        "2 - Errors and warnings",
    ):
        assert numbering_depth(text) is None, text
        assert numbering_tuple(text) is None, text
        assert numbering_prefix_remainder(text) is None, text
        assert heading_rejection_reason(text) == "enumerated_value_item", text
        assert not is_heading_candidate(text), text


def test_space_and_dot_separated_numbers_still_parse_as_outline_numbering():
    for text, tup, depth in (
        ("1 Introduction", (1,), 1),
        ("1. Introduction", (1,), 1),
        ("3.2 Setup", (3, 2), 2),
        ("2 Create a SQL Server credential", (2,), 1),
    ):
        assert numbering_tuple(text) == tup, text
        assert numbering_depth(text) == depth, text
        assert numbering_prefix_remainder(text) is not None, text
        assert heading_rejection_reason(text) != "enumerated_value_item", text
    # The space-separated forms remain heading candidates outright; "1. ..."
    # is still rejected as "list_item" by LIST_LINE_RE, exactly as before.
    for text in ("1 Introduction", "3.2 Setup", "2 Create a SQL Server credential"):
        assert heading_rejection_reason(text) is None, text


# ---------------------------------------------------------------------- #
# looks_like_title_candidate: distinguishes valid headings from good titles

def test_ui_chrome_not_a_title_candidate():
    assert not looks_like_title_candidate("Show answer")
    assert not looks_like_title_candidate("PARTICIPATION")
    assert not looks_like_title_candidate("Animation captions:")
    assert not looks_like_title_candidate("Static figure:")


def test_real_title_is_a_title_candidate():
    assert looks_like_title_candidate("Linear Algebraic Depth-First Search")
    assert looks_like_title_candidate("Design and Analysis of Algorithms")


# ---------------------------------------------------------------------- #
# code/algorithm/equation helper functions directly

def test_looks_like_algorithm_line_is_narrow_and_safe_for_prose():
    assert looks_like_algorithm_line("12: end function")
    assert looks_like_algorithm_line("childCount — 0")
    # Ordinary English prose using "class"/"return" as plain words must NOT
    # trip the narrow, prose-safe variant (only the classifier's fuller
    # variant is allowed to use keyword matching, and only against isolated
    # candidate lines, not continuous prose runs).
    assert not looks_like_algorithm_line("We return to this point in the next class of examples.")


def test_looks_like_code_or_algorithm_line_keyword_bank():
    assert looks_like_code_or_algorithm_line("return -1 // not found")
    assert looks_like_code_or_algorithm_line("def foo():")


def test_looks_like_equation_or_table_row():
    assert looks_like_equation_or_table_row("T=|ss|+1")
    assert looks_like_equation_or_table_row("(3)")
    assert not looks_like_equation_or_table_row("A perfectly normal sentence with no symbols.")


# ---------------------------------------------------------------------- #
# clean_title: publisher/platform names penalized (spec section 11)

def test_clean_title_rejects_platform_names():
    assert clean_title("zyBooks") is None
    assert clean_title("Coursera") is None


def test_clean_title_keeps_real_titles():
    assert clean_title("Linear Algebraic Depth-First Search") == "Linear Algebraic Depth-First Search"


# ---------------------------------------------------------------------- #
# Fix 2: figure/table captions are never headings (document-universal
# convention: "Figure N:"/"Fig. N."/"Table N:", not document vocabulary)

def test_figure_caption_rejected_as_heading():
    assert heading_rejection_reason("Table 1: An LCP array") == "figure_or_table_caption"
    assert heading_rejection_reason("Figure 7: The overall process flow") == "figure_or_table_caption"
    assert heading_rejection_reason("Fig. 3. A sample diagram") == "figure_or_table_caption"


def test_caption_pattern_requires_the_actual_keyword():
    # A numbered section that happens to start with a capitalized word
    # sharing no relation to "figure"/"table" must not be caught.
    assert heading_rejection_reason("3.2 Feature Extraction") is None


# ---------------------------------------------------------------------- #
# Fix 2: bare function-call/notation shape ("KSP(xj, xj)")

def test_bare_function_call_shape_rejected_as_heading():
    assert heading_rejection_reason("KSP(xj, xj)") is not None
    assert looks_like_equation_or_table_row("KSP(xj, xj)")


def test_short_real_heading_with_parens_not_over_rejected():
    # A short descriptive heading that happens to use parens for an
    # abbreviation should not trip the bare-function-call pattern, since
    # that requires the ENTIRE line to be just "identifier(args)".
    assert heading_rejection_reason("Overview") is None


# ---------------------------------------------------------------------- #
# Fix 2: scientific-notation tokenization for numeric-density detection

def test_scientific_notation_split_across_whitespace_counts_as_numeric():
    line = "Spt kernel [24] (3.1 E-3) 3.55 4.0 Crt kernel [24] (1.6 E-3)"
    assert looks_like_equation_or_table_row(line)
    assert heading_rejection_reason(line) is not None


# ---------------------------------------------------------------------- #
# Fix 2: axis/legend label shape (page-region adjacency is the caller's job)

def test_axis_and_legend_label_shapes_detected():
    # Genuine measurement/metric units -- a generic, document-universal set
    # (not tied to any subject matter), unlike "Training"/"Test"/"Data",
    # which are ordinary English words that could just as easily be a real
    # heading ("## Training") and so must NOT be rejected by vocabulary.
    for label in ("Accuracy", "Milliseconds", "Microseconds", "Seconds"):
        assert looks_like_axis_or_legend_label(label), f"{label!r} should look like an axis/legend label"


def test_axis_label_shape_does_not_reject_ordinary_words_by_vocabulary():
    # "Training"/"Test"/"Data" are ordinary words that are legitimate
    # standalone headings in many documents (e.g. an ML paper's "## Training"
    # subsection) -- the axis/legend guard must not treat them as unit
    # words; only real page-region adjacency (checked in pdf_extractor.py,
    # not here) may ever exclude them.
    for word in ("Training", "Test", "Data", "Optimal", "Parameters"):
        assert not looks_like_axis_or_legend_label(word), f"{word!r} must not be treated as a generic unit word"


def test_axis_label_shape_does_not_apply_to_normal_multi_word_headings():
    assert not looks_like_axis_or_legend_label("Design and Analysis of Algorithms")


# ---------------------------------------------------------------------- #
# Fix 3: compound numbered heading decomposition

def test_compound_numbered_heading_splits_into_heading_and_paragraph():
    text = "3.2.3 Suffix Extraction. For efficiency, the first step is to identify boundaries reliably."
    result = split_compound_numbered_heading(text)
    assert result is not None
    num, title, rest = result
    assert num == "3.2.3"
    assert title == "Suffix Extraction"
    assert rest.startswith("For efficiency")


def test_compound_heading_split_ignores_citation_abbreviations():
    text = "3.1 See Smith et al. for details on this topic and its implications for the field."
    assert split_compound_numbered_heading(text) is None


def test_compound_heading_split_ignores_numeric_data_lines():
    text = "3.1 E-3) 3.55 4.0 Crt kernel [24] (1.6 E-3) more numeric data here padding."
    assert split_compound_numbered_heading(text) is None


def test_compound_heading_split_top_level_single_number():
    text = "2 An Example Problem. This section discusses a worked example in full detail for clarity."
    result = split_compound_numbered_heading(text)
    assert result is not None
    num, title, rest = result
    assert num == "2"
    assert title == "An Example Problem"


# ---------------------------------------------------------------------- #
# Fix 3: outline-sequence plausibility

def test_outline_successor_child_and_sibling_transitions():
    assert is_plausible_outline_successor(None, (1,))
    assert is_plausible_outline_successor((3,), (3, 1))       # first child
    assert is_plausible_outline_successor((3, 3), (4,))       # back out to next top-level
    assert is_plausible_outline_successor((3, 1), (3, 2))     # next sibling


def test_outline_successor_rejects_unrelated_jumps_and_repeats():
    assert not is_plausible_outline_successor((3,), (3,))     # exact repeat
    assert not is_plausible_outline_successor((1,), (5, 7))   # unrelated jump


# ---------------------------------------------------------------------- #
# OCR negative-evidence shapes (figure/diagram/table-cell/caption labels):
# these are all still syntactically valid `is_heading_candidate` text (short,
# capitalized, no terminal punctuation) -- the point of these functions is to
# recognize the *shape* so pdf_extractor.py's OCR scorer can weigh it as
# negative evidence, not to duplicate the hard heading_rejection_reason gate.

def test_figure_or_image_label_shapes_detected():
    for text in ("Original Image", "Input Image", "Output Image", "Image 2", "Figure 2", "Sample Photo"):
        assert looks_like_figure_or_diagram_label(text), f"{text!r} should look like a figure/image label"


def test_ordinary_headings_not_treated_as_figure_labels():
    for text in ("Design and Analysis of Algorithms", "Results", "Conclusion", "1 Introduction"):
        assert not looks_like_figure_or_diagram_label(text)


def test_diagram_annotation_shapes_detected():
    for text in ("Y - Axis", "X-Axis", "North Direction Edges", "Color: Red", "Legend:"):
        assert looks_like_diagram_annotation(text), f"{text!r} should look like a diagram annotation"


def test_bare_quantity_label_shape_detected():
    # Structurally identical to a level-1 numbered heading ("1 Introduction"
    # also matches this shape) -- by design, per the function's own
    # docstring: this only tests the syntactic shape, and the caller
    # (pdf_extractor.py's OCR scorer) is responsible for cross-checking
    # outline-sequence plausibility before treating a match as negative
    # evidence, since the ambiguity is real and can't be resolved from the
    # text alone.
    assert looks_like_bare_quantity_label("8 Gray Levels")
    assert looks_like_bare_quantity_label("3 Channels")
    assert looks_like_bare_quantity_label("1 Introduction")
    assert not looks_like_bare_quantity_label("Introduction")


def test_short_introductory_phrase_is_soft_negative_only():
    # Any trailing colon counts (soft signal) -- including on a text that
    # could otherwise be a legitimate heading, since the caller must weigh
    # it against other evidence rather than rejecting unconditionally.
    assert looks_like_short_introductory_phrase("Solution:")
    assert looks_like_short_introductory_phrase("For example:")
    assert looks_like_short_introductory_phrase("Abstract:")


def test_table_cell_fragment_shape_detected():
    assert looks_like_table_cell_fragment("Width: 12")
    assert looks_like_table_cell_fragment("Speed    42.3")


def test_classify_ocr_nonheading_kind_maps_shapes_to_typed_blocks():
    assert classify_ocr_nonheading_kind("Figure 3: An example diagram") == "caption"
    assert classify_ocr_nonheading_kind("Input Image") == "figure_label"
    assert classify_ocr_nonheading_kind("Y - Axis") == "diagram_annotation"
    assert classify_ocr_nonheading_kind("Width: 12") == "table_cell"
    assert classify_ocr_nonheading_kind("Scope") == "unknown"


def test_ocr_heading_negative_signals_collects_all_matches():
    reasons = ocr_heading_negative_signals("Color: Red")
    assert "diagram_annotation" in reasons
    assert ocr_heading_negative_signals("Scope") == []


def test_heading_capitalization_shape_buckets():
    assert heading_capitalization_shape("INTRODUCTION") == "all_caps"
    assert heading_capitalization_shape("Linear Algebraic Depth-First Search") == "title_case"
    assert heading_capitalization_shape("Scope") == "title_case"


# ---------------------------------------------------------------------- #
# ASCII-only numbering digits: Unicode decimal digits (fullwidth '７'
# U+FF17, emitted by icon fonts such as Microsoft Learn's "docons" Note
# glyph) are not section numbers

def test_fullwidth_digit_not_a_numbering_prefix():
    assert numbering_tuple("７ Note") is None
    assert numbering_depth("７ Note") is None


def test_ascii_digit_numbering_prefix_still_parsed():
    assert numbering_tuple("7 Note") == (7,)
    assert numbering_depth("7 Note") == 1


# ---------------------------------------------------------------------- #
# Rejection: "Applies to:" product-badge boilerplate (Microsoft Learn)

def test_applies_to_badge_line_rejected_as_heading():
    assert heading_rejection_reason("Applies to: SQL Server") == "product_badge_line"
    assert heading_rejection_reason("Applies to SQL Server") == "product_badge_line"
    assert heading_rejection_reason("Applies to:") == "product_badge_line"
    assert heading_rejection_reason("APPLIES TO: Azure SQL Database") == "product_badge_line"


def test_ordinary_lines_not_caught_by_badge_rule():
    # Ordinary headings and short colon-ending lines are unaffected by the
    # rule (any existing rejection still applies, but never the badge reason).
    assert is_heading_candidate("Scope")
    assert is_heading_candidate("1 Introduction")
    assert heading_rejection_reason("Solution:") != "product_badge_line"
    # "Applies to" only rejects at the start of the line, and only as a
    # whole word -- a line merely containing the phrase is not a badge row.
    assert heading_rejection_reason("This applies to SQL Server") != "product_badge_line"


# ---------------------------------------------------------------------- #
# Code-keyword gating: a keyword needs code evidence, not just any
# parenthetical or colon, so English headings using those words survive.

def test_english_headings_with_code_keywords_are_not_code():
    for s in ["Import data", "Return values", "Return values (C#)",
              "Import data: overview", "Class diagrams (UML)"]:
        assert not looks_like_code_or_algorithm_line(s), s
        assert is_heading_candidate(s), s


def test_code_lines_with_keywords_still_rejected():
    for s in ["return -1 // not found", "def foo():", "public class Foo",
              "import os", "if (x) > 0:", "return foo(x)"]:
        assert looks_like_code_or_algorithm_line(s), s
        assert not is_heading_candidate(s), s
