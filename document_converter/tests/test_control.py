"""Tests for the top-level control.py ranking module (repo root, not part of
the document_converter package). Relies on pytest's rootdir-based import
mode adding the repo root to sys.path because this tests package sits under
document_converter/, which is itself a package."""
import copy

import pytest

from control import BM25, HybridRanker, cosine_similarity, clean_query


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


def test_clean_query_strips_punctuation_and_lowercases():
    assert clean_query("What's the IP address?!") == "whats the ip address"


def test_bm25_scores_favor_matching_document():
    corpus = [["router", "config"], ["cat", "dog"], ["router", "router", "vlan"]]
    bm25 = BM25(corpus)
    scores = bm25.get_scores(["router"])
    assert scores[2] > scores[0] > scores[1]


def _chunks():
    return [
        {"doc_id": "d1", "doc_title": "Doc1", "chunk_idx": 0, "text": "routing protocols and OSPF details", "embedding": [1, 0, 0]},
        {"doc_id": "d2", "doc_title": "Doc2", "chunk_idx": 0, "text": "completely unrelated cooking recipe", "embedding": [0, 1, 0]},
    ]


def test_ranker_returns_scored_results_in_order():
    chunks = _chunks()
    ranker = HybridRanker(alpha=0.7)
    results = ranker.rank("OSPF routing", chunks, [1, 0, 0], k=2)
    assert results[0]["doc_id"] == "d1"
    assert "score" in results[0]


def test_ranker_does_not_mutate_input_chunk_dicts():
    """Regression test: rank() used to write vector_score/score directly onto
    the shared chunk dicts, which are held long-term by DocStore and shared
    across concurrent search requests. That mutation must not happen."""
    chunks = _chunks()
    snapshot = copy.deepcopy(chunks)
    ranker = HybridRanker(alpha=0.7)
    ranker.rank("OSPF routing", chunks, [1, 0, 0], k=2)
    assert chunks == snapshot
    for c in chunks:
        assert "score" not in c
        assert "vector_score" not in c


def test_ranker_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        HybridRanker(alpha=1.5)


def test_ranker_rejects_nonpositive_k():
    ranker = HybridRanker(alpha=0.5)
    with pytest.raises(ValueError):
        ranker.rank("q", _chunks(), [1, 0, 0], k=0)


def test_ranker_raises_on_empty_chunks():
    ranker = HybridRanker(alpha=0.5)
    with pytest.raises(RuntimeError):
        ranker.rank("q", [], [1, 0, 0], k=2)
