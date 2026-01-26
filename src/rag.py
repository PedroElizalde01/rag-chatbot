from llm import build_prompt, send_prompt
from memory import Memory
from vectorstore import search_vectorstore


def answer_question(question: str, memory: Memory, k: int = 3) -> str:
    memory.add_user(question)
    context = search_vectorstore(question, k=k)
    prompt = build_prompt(context, question, memory.get_history())
    answer = send_prompt(prompt)
    memory.add_assistant(answer)
    return answer
