import google.generativeai as genai


def embed_text(txt: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=txt
    )
    return result["embedding"]
