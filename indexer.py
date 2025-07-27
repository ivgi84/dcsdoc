# indexer.py
"""
Builds a FAISS vector‑store from a PDF and saves it to ./faiss_store/
Run:  python indexer.py  path/to/F-16_manual.pdf
"""
import sys
from pathlib import Path
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chunker import chunk_text_from_pdf


EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR = Path("faiss_store")


def build_index(pdf_path: str | Path) -> None:
    # 1) Split
    chunks = chunk_text_from_pdf(str(pdf_path))

    # 2) Embed & index
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    docs = [Document(page_content=c) for c in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 3) Persist as ./faiss_store/{index + docstore.pkl}
    STORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(STORE_DIR))
    print(f"✅  Vector‑store saved to {STORE_DIR.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python indexer.py path/to/your.pdf"); sys.exit(1)
    build_index(sys.argv[1])
