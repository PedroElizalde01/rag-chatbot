import pytest

import bm25

pytestmark = pytest.mark.unit


def test_search_bm25_empty_index(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25, "BM25_INDEX_PATH", tmp_path / "index.json")
    bm25._CACHED_INDEX = None
    results = bm25.search_bm25("anything", k=3)
    assert results == []
