# chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_pdf


def chunk_text_from_pdf(pdf_path: str, *, size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Loads a PDF and splits it into ~`size`‑char chunks with `overlap` chars overlap.
    """
    raw = load_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(raw)
