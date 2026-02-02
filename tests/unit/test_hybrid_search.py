import pytest

import hybrid_search

pytestmark = pytest.mark.unit


def test_rrf_boosts_documents_in_both_lists():
    list_a = [
        {"id": "d1", "document": "a"},
        {"id": "d2", "document": "b"},
    ]
    list_b = [
        {"id": "d2", "document": "b"},
        {"id": "d1", "document": "a"},
    ]
    fused = hybrid_search.reciprocal_rank_fusion([list_a, list_b], k=2, rrf_k=60)
    assert fused[0]["id"] in {"d1", "d2"}
    assert len(fused) == 2


def test_search_hybrid_returns_documents(monkeypatch):
    monkeypatch.setattr(hybrid_search, "search_vectorstore", lambda query, k: [{"id": "v1", "document": "vec"}])
    monkeypatch.setattr(hybrid_search, "search_bm25", lambda query, k: [{"id": "b1", "document": "bm"}])
    docs = hybrid_search.search_hybrid("q", k=2)
    assert "vec" in docs
    assert "bm" in docs
