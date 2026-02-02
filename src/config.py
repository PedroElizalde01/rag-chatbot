import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))
HYBRID_VECTOR_K = int(os.getenv("HYBRID_VECTOR_K", "8"))
HYBRID_BM25_K = int(os.getenv("HYBRID_BM25_K", "8"))
RRF_K = int(os.getenv("RRF_K", "60"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
