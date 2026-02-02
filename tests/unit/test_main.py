import sys
from pathlib import Path

import pytest

import main

pytestmark = pytest.mark.unit


def test_ensure_documents_loaded_builds_bm25(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "BM25_INDEX_PATH", tmp_path / "bm25.json")
    monkeypatch.setattr(main, "init_vector_store", lambda: None)
    monkeypatch.setattr(main, "ingest_documents_dir", lambda: 0)
    monkeypatch.setattr(main, "refresh_bm25_from_vectorstore", lambda: 3)

    main.ensure_documents_loaded()
    out = capsys.readouterr().out
    assert "No new documents" in out
    assert "Built BM25 index" in out


def test_main_single_query(monkeypatch, capsys):
    monkeypatch.setattr(main, "ensure_documents_loaded", lambda: None)
    monkeypatch.setattr(main, "answer_question", lambda q, m: "resp")
    monkeypatch.setattr(sys, "argv", ["main.py", "what", "is", "x"])

    main.main()
    out = capsys.readouterr().out
    assert "resp" in out


def test_main_interactive_exit(monkeypatch, capsys):
    monkeypatch.setattr(main, "ensure_documents_loaded", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr("builtins.input", lambda _: "")

    main.main()
    out = capsys.readouterr().out
    assert out == ""
