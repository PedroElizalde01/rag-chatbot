import pytest

import vectorstore

pytestmark = pytest.mark.unit


def test_search_vectorstore_formats_results(mock_embeddings, mock_vectorstore):
    mock_vectorstore.add(
        ids=["doc1", "doc2"],
        embeddings=[mock_embeddings("a"), mock_embeddings("b")],
        documents=["A", "B"],
        metadatas=[{"source": "s1"}, {"source": "s2"}],
    )
    results = vectorstore.search_vectorstore("a", k=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert "document" in results[0]
    assert "metadata" in results[0]
    assert results[0]["rank"] == 1
