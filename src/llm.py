import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def build_prompt(context: list[str], question: str) -> str:
    context_text = "\n\n".join(context) if context else ""
    return (
        "You are an expert programming assistant.\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "INSTRUCTIONS:\n"
        "Answer based ONLY on the provided context\n\n"
        "If the answer is not in the context, say \"I do not have that information\"\n\n"
        "Be concise but complete"
    )

def send_prompt(prompt: str, model: str = "models/gemini-flash-latest") -> str:
    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    return response.text or ""
