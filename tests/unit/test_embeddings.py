import pytest

import embeddings

pytestmark = pytest.mark.unit


def test_embed_text_calls_client(monkeypatch):
    def fake_embed_content(model: str, content: str):
        assert model == "models/text-embedding-004"
        assert content == "hello"
        return {"embedding": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(embeddings.genai, "embed_content", fake_embed_content)
    assert embeddings.embed_text("hello") == [0.1, 0.2, 0.3]
