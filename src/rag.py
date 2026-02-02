from llm import build_prompt, send_prompt
from memory import Memory
from hybrid_search import search_hybrid


def answer_question(question: str, memory: Memory, k: int = 3) -> str:
    memory.add_user(question)
    context = search_hybrid(question, k=k)
    prompt = build_prompt(context, question, memory.get_history())
    answer = send_prompt(prompt)
    memory.add_assistant(answer)
    return answer
