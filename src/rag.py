import time

from llm import build_prompt, send_prompt
from memory import Memory
from hybrid_search import search_hybrid

def _estimate_tokens(text: str) -> int:
    return len(text.split())

def answer_question(question: str, memory: Memory, k: int = 3, return_metrics: bool = False):
    if not question or not question.strip():
        raise ValueError("Question is empty")
    start = time.perf_counter()
    memory.add_user(question)
    context = search_hybrid(question, k=k)
    prompt = build_prompt(context, question, memory.get_history())
    answer = send_prompt(prompt)
    memory.add_assistant(answer)
    latency_ms = (time.perf_counter() - start) * 1000
    metrics = {
        "latency_ms": latency_ms,
        "retrieval_k": k,
        "retrieved_count": len(context),
        "prompt_tokens_est": _estimate_tokens(prompt),
    }
    if return_metrics:
        return answer, metrics
    return answer
