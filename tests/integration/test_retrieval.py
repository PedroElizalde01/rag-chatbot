import pytest

import vectorstore

pytestmark = pytest.mark.integration


def test_vector_retrieval_top_k(mock_embeddings, mock_vectorstore):
    mock_vectorstore.add(
        ids=["d1", "d2"],
        embeddings=[mock_embeddings("alpha"), mock_embeddings("beta")],
        documents=["alpha doc", "beta doc"],
        metadatas=[{"source": "s1"}, {"source": "s2"}],
    )
    results = vectorstore.search_vectorstore("alpha", k=1)
    assert results[0]["id"] == "d1"
    assert results[0]["document"] == "alpha doc"
