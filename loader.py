# loader.py
from pathlib import Path
from PyPDF2 import PdfReader


def load_pdf(path: str | Path) -> str:
    """
    Reads a PDF file and returns its text as a single string.
    Non‑text pages (scans) are silently skipped.
    """
    reader = PdfReader(str(path))
    return "\n".join(
        page.extract_text() or ""   # extract_text may return None
        for page in reader.pages
    )
