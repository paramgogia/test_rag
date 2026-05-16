"""
FAISS-backed vector store, with embeddings from Gemini's text-embedding-004.

We batch embedding requests (Gemini allows up to 100 per call) and persist
the index + chunk metadata to disk so ingestion only runs once.
"""
from __future__ import annotations
import pickle
import time
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import google.generativeai as genai

from config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH, GEMINI_API_KEY, EMBEDDING_DIM
)
from document_loader import Chunk


EMBEDDING_DIM = 768  # text-embedding-004 output dim
EMBED_BATCH = 10   # safely below Gemini's 100/req limit


def _ensure_api_key():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=GEMINI_API_KEY)


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Embed a list of texts. task_type="RETRIEVAL_DOCUMENT" for ingestion,
    "RETRIEVAL_QUERY" at query time -- Gemini optimises the vectors
    differently for each.
    """
    _ensure_api_key()
    all_vectors = []
    total_batches = (len(texts) + EMBED_BATCH - 1) // EMBED_BATCH

    for batch_idx, i in enumerate(range(0, len(texts), EMBED_BATCH), start=1):
        batch = texts[i:i + EMBED_BATCH]
        # Retry loop with awareness of server-provided retry_delay
        for attempt in range(5):
            try:
                resp = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                )
                vecs = resp["embedding"]
                if isinstance(vecs[0], (int, float)):
                    vecs = [vecs]
                all_vectors.extend(vecs)
                print(f"    Batch {batch_idx}/{total_batches} embedded ({len(all_vectors)}/{len(texts)} done)")
                break
            except Exception as e:
                msg = str(e)
                # Honour server-provided retry hint when rate-limited
                wait = 35  # default backoff for 429
                m = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                if m:
                    wait = float(m.group(1)) + 2
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    if attempt == 4:
                        raise
                    print(f"    Rate-limited, waiting {wait:.0f}s before retry...")
                    time.sleep(wait)
                else:
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))

        # Proactive pacing between batches: ~7 seconds keeps us safely
        # under 100 texts/min at batch size 10
        if batch_idx < total_batches:
            time.sleep(7)

    arr = np.array(all_vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

class VectorStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []

    # ----- persistence -----
    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump([c.to_dict() for c in self.chunks], f)

    def load(self) -> bool:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            return False
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks = [Chunk(**d) for d in data]
        return True

    # ----- build -----
    def build(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index.")
        print(f"  Embedding {len(chunks)} chunks (in batches of {EMBED_BATCH})...")
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.chunks = chunks
        for i, c in enumerate(self.chunks):
            c.chunk_id = i
        self.save()
        print(f"  Vector store built. Total vectors: {self.index.ntotal}")

    # ----- search -----
    def search(self, query: str, top_k: int = 6) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Vector store not loaded. Run ingest.py first.")
        q_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
        scores, idxs = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
