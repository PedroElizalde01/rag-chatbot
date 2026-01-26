from config import MEMORY_MAX_MESSAGES


class Memory:
    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES):
        self.max_messages = max_messages
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str) -> None:
        self._add("user", text)

    def add_assistant(self, text: str) -> None:
        self._add("assistant", text)

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []

    def _add(self, role: str, text: str) -> None:
        self.messages.append({"role": role, "content": text})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
