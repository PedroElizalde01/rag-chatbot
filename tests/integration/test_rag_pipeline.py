import pytest

import rag
from memory import Memory

pytestmark = pytest.mark.integration


def test_answer_question_returns_metrics(monkeypatch):
    monkeypatch.setattr(rag, "search_hybrid", lambda q, k: ["ctx1", "ctx2"])
    captured = {}

    def fake_send(prompt: str) -> str:
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(rag, "send_prompt", fake_send)
    memory = Memory(max_messages=3)
    answer, metrics = rag.answer_question("What is FastAPI?", memory, k=2, return_metrics=True)
    assert answer == "ok"
    assert "What is FastAPI?" in captured["prompt"]
    assert "Source 1:" in captured["prompt"]
    assert "ctx1" in captured["prompt"]
    assert metrics["latency_ms"] >= 0
    assert metrics["retrieval_k"] == 2
    assert metrics["retrieved_count"] == 2
    assert metrics["prompt_tokens_est"] > 0


def test_answer_question_empty_raises():
    memory = Memory()
    import pytest

    with pytest.raises(ValueError):
        rag.answer_question("   ", memory)


def test_answer_question_llm_failure_propagates(monkeypatch):
    monkeypatch.setattr(rag, "search_hybrid", lambda q, k: ["ctx"])

    def raise_timeout(_):
        raise TimeoutError("llm timeout")

    monkeypatch.setattr(rag, "send_prompt", raise_timeout)
    memory = Memory()
    import pytest

    with pytest.raises(TimeoutError):
        rag.answer_question("hi", memory)
