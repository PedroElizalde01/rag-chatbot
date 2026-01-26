from llm import build_prompt, send_prompt
from vectorstore import search_vectorstore


def answer_question(question: str, k: int = 3) -> str:
    context = search_vectorstore(question, k=k)
    prompt = build_prompt(context, question)
    return send_prompt(prompt)
