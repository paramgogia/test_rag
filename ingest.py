"""
Run this once after dropping documents into ./data/.

Reads every supported document, splits it into chunks, embeds them with
Gemini, and writes a FAISS index + metadata to ./vectorstore/.

Usage:
    python ingest.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from config import DATA_DIR, SOURCE_FILES, CHUNK_SIZE, CHUNK_OVERLAP
from document_loader import load_document
from vector_store import VectorStore


def main():
    print("=" * 60)
    print("Infosys RAG - Document Ingestion")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found at {DATA_DIR}")
        sys.exit(1)

    all_chunks = []
    for source_name, file_name in SOURCE_FILES.items():
        path = DATA_DIR / file_name
        if not path.exists():
            print(f"  [skip] {file_name} not found in data/")
            continue
        print(f"\n--> Loading {source_name} ({file_name})")
        try:
            chunks = load_document(path, source_name, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"    Produced {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"    ERROR loading {file_name}: {e}")

    if not all_chunks:
        print("\nNo chunks generated. Make sure your documents are in ./data/")
        sys.exit(1)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    print("\nBuilding vector store with Gemini embeddings...")
    store = VectorStore()
    store.build(all_chunks)
    print("\nDone. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
