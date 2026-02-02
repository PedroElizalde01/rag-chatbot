import pytest

import ingestion

pytestmark = pytest.mark.integration


def test_ingest_pdf_batches_and_updates_bm25(monkeypatch):
    monkeypatch.setattr(ingestion, "read_pdf", lambda path: "text")
    monkeypatch.setattr(ingestion, "chunk_text", lambda text, chunk_size, chunk_overlap: ["c1", "c2"])
    monkeypatch.setattr(ingestion, "embed_text", lambda text: [1.0, 0.0])

    added = {}

    class FakeCollection:
        def add(self, ids, embeddings, documents, metadatas):
            added["ids"] = ids
            added["documents"] = documents
            added["metadatas"] = metadatas

    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: FakeCollection())

    upserted = {}

    def fake_upsert(ids, documents, metadatas):
        upserted["ids"] = ids
        upserted["documents"] = documents
        upserted["metadatas"] = metadatas

    monkeypatch.setattr(ingestion, "upsert_bm25_documents", fake_upsert)

    count = ingestion.ingest_pdf("/tmp/sample.pdf")
    assert count == 2
    assert added["ids"] == ["sample_chunk_0", "sample_chunk_1"]
    assert upserted["ids"] == ["sample_chunk_0", "sample_chunk_1"]


def test_ingest_pdf_missing_file_propagates(monkeypatch):
    def raise_missing(_):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ingestion, "read_pdf", raise_missing)
    import pytest

    with pytest.raises(FileNotFoundError):
        ingestion.ingest_pdf("/tmp/missing.pdf")
