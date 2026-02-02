import google.generativeai as genai
from config import GEMINI_MODEL, GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def build_prompt(context: list[str], question: str, history: list[dict[str, str]]) -> str:
    if context:
        context_text = "\n\n".join(
            f"Source {idx + 1}: {chunk}" for idx, chunk in enumerate(context)
        )
    else:
        context_text = ""
    history_text = "\n".join(
        f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}"
        for item in history
    )
    return (
        "You are an expert programming assistant.\n\n"
        "CONVERSATION HISTORY:\n"
        f"{history_text}\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "INSTRUCTIONS:\n"
        "Answer based ONLY on the provided context\n\n"
        "If the answer is not in the context, say \"I do not have that information\"\n\n"
        "Be concise but complete"
    )

def send_prompt(prompt: str, model: str = GEMINI_MODEL) -> str:
    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    return response.text or ""
