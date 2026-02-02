import pytest

from memory import Memory

pytestmark = pytest.mark.unit


def test_memory_trims_to_max():
    memory = Memory(max_messages=2)
    memory.add_user("a")
    memory.add_assistant("b")
    memory.add_user("c")
    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["content"] == "b"
    assert history[1]["content"] == "c"
