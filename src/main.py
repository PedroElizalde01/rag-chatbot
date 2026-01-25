import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from embeddings import embed_text

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    dog = embed_text("dog")
    cat = embed_text("cat")
    car = embed_text("car")
    
    print(f"Similarity dog-cat: {cosine_similarity(dog, cat):.4f}")
    print(f"Similarity dog-car: {cosine_similarity(dog, car):.4f}")
    print(f"Similarity cat-car: {cosine_similarity(cat, car):.4f}")


if __name__ == "__main__":
    main()
