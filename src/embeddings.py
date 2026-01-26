import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def embed_text(txt: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=txt
    )
    return result["embedding"]
