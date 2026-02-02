from pathlib import Path

import pytest

import bm25

pytestmark = pytest.mark.unit


def test_rebuild_and_search_bm25(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25, "BM25_INDEX_PATH", tmp_path / "index.json")
    bm25._CACHED_INDEX = None
    bm25.rebuild_bm25_index(
        ids=["d1", "d2"],
        documents=["fastapi framework", "robotics"],
        metadatas=[{"s": 1}, {"s": 2}],
    )
    results = bm25.search_bm25("fastapi", k=1)
    assert results[0]["id"] == "d1"
    assert Path(bm25.BM25_INDEX_PATH).exists()


def test_upsert_updates_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(bm25, "BM25_INDEX_PATH", tmp_path / "index.json")
    bm25._CACHED_INDEX = None
    bm25.rebuild_bm25_index(
        ids=["d1"],
        documents=["hello world"],
        metadatas=[{"v": 1}],
    )
    bm25.upsert_bm25_documents(
        ids=["d1", "d2"],
        documents=["updated hello", "new doc"],
        metadatas=[{"v": 2}, {"v": 3}],
    )
    index = bm25.get_bm25_index()
    assert len(index.ids) == 2
    results = bm25.search_bm25("updated", k=1)
    assert results[0]["id"] == "d1"
