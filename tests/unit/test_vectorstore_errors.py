import pytest

import vectorstore

pytestmark = pytest.mark.unit


def test_search_vectorstore_propagates_embedding_error(monkeypatch):
    def raise_embed(_):
        raise RuntimeError("embed failure")

    monkeypatch.setattr(vectorstore, "embed_text", raise_embed)
    monkeypatch.setattr(vectorstore, "get_or_create_collection", lambda: None)
    import pytest

    with pytest.raises(RuntimeError):
        vectorstore.search_vectorstore("q")


def test_search_vectorstore_propagates_collection_error(monkeypatch):
    class BrokenCollection:
        def query(self, **kwargs):
            raise ConnectionError("down")

    monkeypatch.setattr(vectorstore, "get_or_create_collection", lambda: BrokenCollection())
    monkeypatch.setattr(vectorstore, "embed_text", lambda text: [0.1, 0.2])
    import pytest

    with pytest.raises(ConnectionError):
        vectorstore.search_vectorstore("q")
