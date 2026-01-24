from doc_loader import read_pdf, chunk_text

def main():
    pdf_path = "data/fastapi.pdf"
    pdf_content = read_pdf(pdf_path)
    chunks = chunk_text(pdf_content, chunk_size=500, chunk_overlap=50)
    for chunk in chunks:
        print(chunk)
        print("-" * 100)

if __name__ == "__main__":
    main()