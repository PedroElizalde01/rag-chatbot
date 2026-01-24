from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def chunk_text(txt: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="")
    return text_splitter.split_text(txt)