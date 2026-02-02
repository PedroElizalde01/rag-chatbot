import pytest

import vectorstore

pytestmark = pytest.mark.unit


def test_get_or_create_collection(monkeypatch):
    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_or_create_collection(self, name):
            return {"name": name}

    monkeypatch.setattr(vectorstore, "get_chroma_client", lambda: FakeClient("/tmp"))
    result = vectorstore.get_or_create_collection("docs")
    assert result == {"name": "docs"}
