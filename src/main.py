import sys

from vectorstore import search_vectorstore

def main():
    query = " ".join(sys.argv[1:]) or "fastapi"
    results = search_vectorstore(query)
    for i, chunk in enumerate(results, start=1):
        print(f"{i}. {chunk}")


if __name__ == "__main__":
    main()
