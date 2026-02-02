from pathlib import Path

import pytest

import ingestion

pytestmark = pytest.mark.unit


def test_init_vector_store_creates_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "VECTORDB_PATH", tmp_path / "vectordb")
    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: "collection")

    result = ingestion.init_vector_store()
    assert result == "collection"
    assert (tmp_path / "vectordb").exists()


def test_is_pdf_ingested_true(monkeypatch):
    class FakeCollection:
        def get(self, ids):
            return {"ids": ["doc_chunk_0"]}

    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: FakeCollection())
    assert ingestion.is_pdf_ingested("/tmp/doc.pdf") is True


def test_is_pdf_ingested_false_on_error(monkeypatch):
    class FakeCollection:
        def get(self, ids):
            raise RuntimeError("missing")

    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: FakeCollection())
    assert ingestion.is_pdf_ingested("/tmp/doc.pdf") is False


def test_ingest_documents_dir_skips_ingested(monkeypatch, tmp_path):
    pdf1 = tmp_path / "a.pdf"
    pdf2 = tmp_path / "b.pdf"
    pdf1.write_text("x")
    pdf2.write_text("y")

    monkeypatch.setattr(ingestion, "is_pdf_ingested", lambda path: Path(path).stem == "a")
    monkeypatch.setattr(ingestion, "ingest_pdf", lambda path: 2)

    total = ingestion.ingest_documents_dir(tmp_path)
    assert total == 2


def test_refresh_bm25_from_vectorstore(monkeypatch):
    class FakeCollection:
        def get(self, include):
            return {
                "ids": ["d1"],
                "documents": ["doc"],
                "metadatas": [{"s": 1}],
            }

    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: FakeCollection())
    monkeypatch.setattr(ingestion, "rebuild_bm25_index", lambda ids, documents, metadatas: len(ids))

    assert ingestion.refresh_bm25_from_vectorstore() == 1


def test_refresh_bm25_from_vectorstore_empty(monkeypatch):
    class FakeCollection:
        def get(self, include):
            return {"ids": [], "documents": [], "metadatas": []}

    monkeypatch.setattr(ingestion, "get_or_create_collection", lambda: FakeCollection())
    assert ingestion.refresh_bm25_from_vectorstore() == 0
