import sys

from rag import answer_question

def main():
    query = " ".join(sys.argv[1:]) or "fastapi"
    print(answer_question(query))


if __name__ == "__main__":
    main()
