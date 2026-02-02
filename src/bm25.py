import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = Path(__file__).parent.parent / "data" / "bm25" / "index.json"

_LOCK = RLock()
_CACHED_INDEX = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    _bm25: BM25Okapi | None

    @classmethod
    def build(cls, ids: list[str], documents: list[str], metadatas: list[dict]) -> "BM25Index":
        tokenized = [_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        return cls(ids=ids, documents=documents, metadatas=metadatas, _bm25=bm25)

    def search(self, query: str, k: int) -> list[dict]:
        if not self.documents or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append(
                {
                    "id": self.ids[idx],
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(score),
                    "rank": rank,
                }
            )
        return results


def _load_index() -> BM25Index:
    if not BM25_INDEX_PATH.exists():
        return BM25Index.build([], [], [])
    with BM25_INDEX_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items", [])
    ids = [item["id"] for item in items]
    documents = [item["document"] for item in items]
    metadatas = [item.get("metadata", {}) for item in items]
    return BM25Index.build(ids, documents, metadatas)


def _save_index(index: BM25Index) -> None:
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for doc_id, doc, metadata in zip(index.ids, index.documents, index.metadatas):
        items.append({"id": doc_id, "document": doc, "metadata": metadata})
    payload = {"items": items}
    with BM25_INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def get_bm25_index() -> BM25Index:
    global _CACHED_INDEX
    with _LOCK:
        if _CACHED_INDEX is None:
            _CACHED_INDEX = _load_index()
        return _CACHED_INDEX


def rebuild_bm25_index(ids: list[str], documents: list[str], metadatas: list[dict]) -> int:
    global _CACHED_INDEX
    with _LOCK:
        _CACHED_INDEX = BM25Index.build(ids, documents, metadatas)
        _save_index(_CACHED_INDEX)
        return len(ids)


def upsert_bm25_documents(ids: list[str], documents: list[str], metadatas: list[dict]) -> int:
    global _CACHED_INDEX
    with _LOCK:
        index = get_bm25_index()
        existing = {doc_id: i for i, doc_id in enumerate(index.ids)}
        for doc_id, doc, metadata in zip(ids, documents, metadatas):
            if doc_id in existing:
                idx = existing[doc_id]
                index.documents[idx] = doc
                index.metadatas[idx] = metadata
            else:
                index.ids.append(doc_id)
                index.documents.append(doc)
                index.metadatas.append(metadata)
        _CACHED_INDEX = BM25Index.build(index.ids, index.documents, index.metadatas)
        _save_index(_CACHED_INDEX)
        return len(ids)


def search_bm25(query: str, k: int = 3) -> list[dict]:
    index = get_bm25_index()
    return index.search(query, k)
