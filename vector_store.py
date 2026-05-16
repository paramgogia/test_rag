"""
FAISS-backed vector store with pluggable embeddings.

Supports two providers, chosen via config.LLM_PROVIDER:
- "gemini": Google's gemini-embedding-001 (cloud, subject to free-tier quotas)
- "ollama": nomic-embed-text running locally (no quotas, fully offline)
"""
from __future__ import annotations
import pickle
import re
import time
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import (
    LLM_PROVIDER,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, OLLAMA_EMBEDDING_DIM,
    FAISS_INDEX_PATH, METADATA_PATH, GEMINI_API_KEY
)
from document_loader import Chunk


# Pick embedding dim based on provider
if LLM_PROVIDER == "ollama":
    ACTIVE_EMBEDDING_DIM = OLLAMA_EMBEDDING_DIM
else:
    ACTIVE_EMBEDDING_DIM = EMBEDDING_DIM

EMBED_BATCH = 32  # tuned for local; bumped up since there's no quota
# nomic-embed-text has an 8k token window but Ollama's default num_ctx is 2048.
# ~4000 chars stays comfortably under 1024 tokens for any input.
OLLAMA_EMBED_MAX_CHARS = 4000


# ---------- Provider-specific embedding functions ----------

def _embed_gemini(texts: List[str], task_type: str) -> List[List[float]]:
    import google.generativeai as genai
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Either set it in .env or switch "
            "LLM_PROVIDER to 'ollama' in config.py."
        )
    genai.configure(api_key=GEMINI_API_KEY)

    all_vectors = []
    for i in range(0, len(texts), 10):  # smaller batch for Gemini rate limits
        batch = texts[i:i + 10]
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
                break
            except Exception as e:
                msg = str(e)
                wait = 35
                m = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                if m:
                    wait = float(m.group(1)) + 2
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if attempt == 4:
                        raise
                    print(f"    Rate-limited, waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
        if i + 10 < len(texts):
            time.sleep(7)
    return all_vectors


def _ollama_embed_call(client, inputs: List[str]) -> List[List[float]]:
    """One embed call with explicit truncation, returning a list of vectors."""
    resp = client.embed(
        model=OLLAMA_EMBEDDING_MODEL,
        input=inputs,
        truncate=True,
    )
    vecs = resp.get("embeddings") or resp.get("embedding")
    if vecs is None:
        raise RuntimeError(f"Unexpected Ollama response: {resp}")
    if isinstance(vecs[0], (int, float)):
        vecs = [vecs]
    return vecs


def _embed_ollama(texts: List[str], _task_type: str) -> List[List[float]]:
    """
    Embed using a local Ollama server. The Ollama embed endpoint can take
    a list of inputs in one call, which makes it nicely batch-friendly.

    Safeguards:
    - Pre-truncate each input to OLLAMA_EMBED_MAX_CHARS so a single oversize
      chunk can't blow the embedder's context window.
    - On a batch failure, retry each item individually so one bad input does
      not poison the whole batch.
    """
    import ollama
    client = ollama.Client(host=OLLAMA_HOST)

    # Pre-truncate every text once; embedding doesn't need the full chunk,
    # and nomic-embed-text's effective context is small.
    safe_texts = [t[:OLLAMA_EMBED_MAX_CHARS] for t in texts]

    all_vectors: List[List[float]] = []
    for i in range(0, len(safe_texts), EMBED_BATCH):
        batch = safe_texts[i:i + EMBED_BATCH]
        vecs = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                vecs = _ollama_embed_call(client, batch)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        if vecs is None:
            # Batch failed repeatedly — fall back to one-by-one with aggressive
            # truncation, so a single oversize input can't sink the whole run.
            print(f"    batch {i}-{i + len(batch)} failed ({last_err}); "
                  f"retrying inputs individually")
            vecs = []
            for j, item in enumerate(batch):
                item_text = item
                ok = False
                for shrink in (1.0, 0.5, 0.25):
                    trimmed = item_text[: max(200, int(OLLAMA_EMBED_MAX_CHARS * shrink))]
                    try:
                        vecs.extend(_ollama_embed_call(client, [trimmed]))
                        ok = True
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(1.0)
                if not ok:
                    raise RuntimeError(
                        f"Ollama embedding failed on chunk {i + j} even after "
                        f"truncation. Is the Ollama app running and is "
                        f"'{OLLAMA_EMBEDDING_MODEL}' pulled? "
                        f"Run: ollama pull {OLLAMA_EMBEDDING_MODEL}\n"
                        f"Original error: {last_err}"
                    )

        all_vectors.extend(vecs)
        done = min(i + EMBED_BATCH, len(safe_texts))
        print(f"    {done}/{len(safe_texts)} embedded")
    return all_vectors


def embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Top-level embedding entry point. Routes to the configured provider,
    L2-normalises the resulting vectors so FAISS inner-product == cosine.
    """
    if not texts:
        return np.zeros((0, ACTIVE_EMBEDDING_DIM), dtype="float32")

    if LLM_PROVIDER == "ollama":
        all_vectors = _embed_ollama(texts, task_type)
    elif LLM_PROVIDER == "gemini":
        all_vectors = _embed_gemini(texts, task_type)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    arr = np.array(all_vectors, dtype="float32")
    if arr.shape[1] != ACTIVE_EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dim mismatch: model returned {arr.shape[1]}, "
            f"config expects {ACTIVE_EMBEDDING_DIM}. "
            f"Update {('OLLAMA_EMBEDDING_DIM' if LLM_PROVIDER == 'ollama' else 'EMBEDDING_DIM')} in config.py."
        )
    faiss.normalize_L2(arr)
    return arr


# ---------- Vector store class (unchanged interface) ----------

class VectorStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []

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

    def build(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index.")
        print(f"  Embedding {len(chunks)} chunks via {LLM_PROVIDER}...")
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

        self.index = faiss.IndexFlatIP(ACTIVE_EMBEDDING_DIM)
        self.index.add(vectors)
        self.chunks = chunks
        for i, c in enumerate(self.chunks):
            c.chunk_id = i
        self.save()
        print(f"  Vector store built. Total vectors: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 6) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for the query.

        To stop the much-larger Annual Report from drowning out short
        press releases / sheets in pure cosine ranking, we:
        1. Pull a wider candidate pool (4x top_k) from FAISS.
        2. Boost candidates whose source name matches tokens in the query
           (e.g. "Q1 FY26" in the query lifts "Q1 FY26 Press Release").
        3. Re-rank by boosted score and return top_k.
        """
        if self.index is None:
            raise RuntimeError("Vector store not loaded. Run ingest.py first.")
        q_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")

        pool = min(max(top_k * 4, top_k + 10), self.index.ntotal)
        scores, idxs = self.index.search(q_vec, pool)

        q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 2}

        rescored: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            meta = f"{chunk.source} {chunk.location}".lower()
            meta_tokens = set(re.findall(r"[a-z0-9]+", meta))
            overlap = len(q_tokens & meta_tokens)
            # Small boost per overlapping token, capped so it can re-rank close
            # scores (gap is typically <0.05) without overwhelming similarity.
            boost = min(0.04 * overlap, 0.12)
            rescored.append((chunk, float(score) + boost))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]