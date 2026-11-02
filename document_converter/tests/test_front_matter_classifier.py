"""Unit tests for front_matter.py's line classifier (spec section 4). Pure
text-shape tests, no PDF/OCR involved."""
from document_converter.front_matter import classify_front_matter_line


def test_author_line_shapes():
    assert classify_front_matter_line("Jane A. Doe") == "author"
    assert classify_front_matter_line("John Smith and Jane Doe") == "author"
    assert classify_front_matter_line("Jane Doe1, John Smith2") == "author"


def test_affiliation_and_contact():
    assert classify_front_matter_line("Carnegie Mellon University") == "affiliation"
    assert classify_front_matter_line("Department of Computer Science, Stanford University") == "affiliation"
    assert classify_front_matter_line("jane.doe@cmu.edu") == "contact"


def test_copyright_permission_conference():
    assert classify_front_matter_line(
        "Permission to make digital or hard copies of part of this work is granted."
    ) == "permission"
    assert classify_front_matter_line("Copyright 2024 ACM. All rights reserved.") == "copyright"
    assert classify_front_matter_line(
        "Proceedings of the 30th Symposium on Operating Systems Principles"
    ) == "conference"
    assert classify_front_matter_line("ISBN 978-1-4503-1234-5") == "publication_metadata"


def test_keywords_label():
    assert classify_front_matter_line("Keywords: computer systems, education") == "keywords"


def test_ordinary_body_prose_not_classified():
    assert classify_front_matter_line(
        "Unfortunately, most computer science and computer engineering curricula do not have a course."
    ) is None
    assert classify_front_matter_line("With the ever-increasing use of the Internet.") is None


def test_venue_line_without_a_phrase_bank_match_recognized_by_date_tail():
    """A publication-venue citation line naming a journal/society/
    transactions series the fixed phrase bank can't enumerate is still
    recognized generically by its near-universal closing shape: "<venue>,
    <City>, <Country>, <Month> <Day>[-<Day>], <Year>"."""
    assert classify_front_matter_line(
        "Transactions of the Korean Nuclear Society Spring Meeting Jeju, Korea, May 17-18, 2018"
    ) == "conference"
    assert classify_front_matter_line(
        "International Workshop on Widget Design, Springfield, USA, March 3-4, 2021"
    ) == "conference"


def test_compound_author_line_with_glued_address_and_corresponding_author():
    """OCR sometimes glues author names directly onto a trailing mailing
    address and corresponding-author marker with no line break between
    fields -- the whole compound line must still classify as "author" so it
    is grouped as front matter, not left as an unclassified body
    paragraph, even though the line as a whole is far longer and far less
    name-shaped than an ordinary author byline."""
    assert classify_front_matter_line(
        'Yong Suk Suh", Seung Ki Shin, Dane Baang, Sang Mun Seo Yuseong-Gu, '
        'Daejeon, 34057, Korea "Corresponding author: yssuh@kaeri.re.kr'
    ) == "author"
    assert classify_front_matter_line(
        "Jane A. Doe, John Q. Public, Main Street, Springfield, 62704, USA "
        "Corresponding author: jane.doe@example.edu"
    ) == "author"


def test_single_capitalized_word_starting_a_sentence_not_treated_as_compound_author():
    """The compound-author fallback requires >=2 leading name-shaped groups
    plus independent corroborating tail evidence -- a single capitalized
    word starting an ordinary long sentence must not be enough by itself,
    even if the sentence happens to be long."""
    assert classify_front_matter_line(
        "Stanford researchers published a new result last year that significantly "
        "improved on the previous state of the art in this specific area of study."
    ) is None
