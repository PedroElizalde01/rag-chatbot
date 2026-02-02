import pytest

import loader

pytestmark = pytest.mark.unit


def test_read_pdf_concatenates_pages(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("a"), FakePage("b")]

    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    assert loader.read_pdf("fake.pdf") == "a\nb"


def test_read_pdf_propagates_error(monkeypatch):
    class FakeReader:
        def __init__(self, path):
            raise ValueError("invalid pdf")

    monkeypatch.setattr(loader, "PdfReader", FakeReader)
    import pytest

    with pytest.raises(ValueError):
        loader.read_pdf("invalid.txt")


def test_chunk_text_overlap():
    chunks = loader.chunk_text("abcdef", chunk_size=3, chunk_overlap=1)
    assert all(chunks)
    assert all(len(chunk) <= 3 for chunk in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-1] == nxt[0]
