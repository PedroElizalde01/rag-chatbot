import pytest

import llm

pytestmark = pytest.mark.unit


def test_build_prompt_no_context():
    prompt = llm.build_prompt([], "Q", [])
    assert "CONTEXT:" in prompt
    assert "Source" not in prompt
