import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))
