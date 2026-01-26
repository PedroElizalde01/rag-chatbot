import sys

from memory import Memory
from rag import answer_question

def main():
    memory = Memory()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(answer_question(query, memory))
        return

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        print(answer_question(query, memory))


if __name__ == "__main__":
    main()
