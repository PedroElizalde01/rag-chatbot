import pytest

import llm

pytestmark = pytest.mark.unit


def test_build_prompt_contains_sections():
    prompt = llm.build_prompt(["context chunk"], "What is X?", [{"role": "user", "content": "Hi"}])
    assert "CONTEXT:" in prompt
    assert "Source 1:" in prompt
    assert "context chunk" in prompt
    assert "USER QUESTION:" in prompt
    assert "What is X?" in prompt
    assert "CONVERSATION HISTORY:" in prompt


def test_send_prompt_returns_text(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt: str):
            return FakeResponse("ok")

    monkeypatch.setattr(llm.genai, "GenerativeModel", FakeModel)
    assert llm.send_prompt("hello") == "ok"


def test_send_prompt_handles_empty_text(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, prompt: str):
            return FakeResponse(None)

    monkeypatch.setattr(llm.genai, "GenerativeModel", FakeModel)
    assert llm.send_prompt("hello") == ""
