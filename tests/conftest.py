import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

import embeddings
import llm
import vectorstore


def _deterministic_vector(text: str) -> list[float]:
    base = sum(ord(ch) for ch in text) or 1
    return [base % 7 / 7, base % 11 / 11, base % 13 / 13]


class InMemoryCollection:
    def __init__(self):
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str]):
        query = query_embeddings[0]
        scored = []
        for idx, emb in enumerate(self.embeddings):
            score = -sum((q - e) ** 2 for q, e in zip(query, emb))
            scored.append((idx, score))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:n_results]
        ids = [self.ids[i] for i, _ in ranked]
        docs = [self.documents[i] for i, _ in ranked]
        metas = [self.metadatas[i] for i, _ in ranked]
        dists = [1.0 - score for _, score in ranked]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    def get(self, include: list[str]):
        return {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }


@pytest.fixture()
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def mock_embeddings(monkeypatch):
    def _embed(text: str) -> list[float]:
        return _deterministic_vector(text)

    monkeypatch.setattr(embeddings, "embed_text", _embed)
    monkeypatch.setattr(vectorstore, "embed_text", _embed)
    return _embed


@pytest.fixture()
def mock_llm(monkeypatch):
    def _send(prompt: str) -> str:
        return "mock-answer"

    monkeypatch.setattr(llm, "send_prompt", _send)
    return _send


@pytest.fixture()
def mock_vectorstore(monkeypatch):
    collection = InMemoryCollection()
    monkeypatch.setattr(vectorstore, "get_or_create_collection", lambda name="documents": collection)
    return collection
