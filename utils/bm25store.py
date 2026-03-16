"""
BM25 Store — manages one BM25 index per uploaded document.

Why per-file?
  - We need to filter by filename (just like ChromaDB's where filter)
  - One global BM25 index would mix chunks from all documents
  - Per-file index means: search panda.pdf BM25 separately from XAI.pdf BM25

Storage: in-memory dict  {filename: {"chunks": [...], "bm25": BM25Okapi}}
Persists across requests (module-level singleton) but resets on server restart.
For production: serialize to disk with pickle.
"""

import pickle
import os
from rank_bm25 import BM25Okapi

# ── In-memory store ───────────────────────────────────────────────────────────
_bm25_indexes: dict = {}  

# ── Optional disk persistence path ───────────────────────────────────────────
BM25_PERSIST_PATH = "bm25_indexes.pkl"


def _save_to_disk():
    """Save indexes to disk so they survive server restarts."""
    try:
        with open(BM25_PERSIST_PATH, "wb") as f:
            pickle.dump(_bm25_indexes, f)
    except Exception as e:
        print(f"[BM25] Warning: could not save to disk: {e}")


def _load_from_disk():
    """Load indexes from disk on startup."""
    global _bm25_indexes
    if os.path.exists(BM25_PERSIST_PATH):
        try:
            with open(BM25_PERSIST_PATH, "rb") as f:
                _bm25_indexes = pickle.load(f)
            print(f"[BM25] Loaded {len(_bm25_indexes)} indexes from disk")
        except Exception as e:
            print(f"[BM25] Warning: could not load from disk: {e}")
            _bm25_indexes = {}


# Load on import
_load_from_disk()


def add_to_bm25_index(filename: str, chunks: list):
    """
    Add a list of text chunks to the BM25 index for a given filename.
    Called during document ingestion.
    Appends to existing index if file was already indexed (re-upload).
    """
    existing_chunks = []

    if filename in _bm25_indexes:
        existing_chunks = _bm25_indexes[filename]["chunks"]

    all_chunks = existing_chunks + [c for c in chunks if c not in existing_chunks]

    tokenized = [chunk.lower().split() for chunk in all_chunks]

    _bm25_indexes[filename] = {
        "chunks": all_chunks,
        "bm25": BM25Okapi(tokenized),
    }

    print(f"[BM25] Index updated for '{filename}': {len(all_chunks)} chunks total")
    _save_to_disk()


def bm25_search(query: str, filenames: list, top_k: int = 4) -> list:
    """
    Search BM25 indexes for the given filenames.
    Returns top_k chunks ranked by BM25 score across all specified files.

    Args:
        query:     the search query
        filenames: list of filenames to search (matches ChromaDB's $in filter)
        top_k:     number of top results to return

    Returns:
        list of (chunk_text, score, filename) tuples sorted by score desc
    """
    if not filenames:
        return []

    tokenized_query = query.lower().split()
    all_scored = []

    for filename in filenames:
        if filename not in _bm25_indexes:
            print(f"[BM25] No index found for '{filename}' — skipping")
            continue

        index_data = _bm25_indexes[filename]
        chunks = index_data["chunks"]
        bm25   = index_data["bm25"]

        scores = bm25.get_scores(tokenized_query)

        for chunk, score in zip(chunks, scores):
            if score > 0:
                all_scored.append((chunk, float(score), filename))

    all_scored.sort(key=lambda x: x[1], reverse=True)

    print(f"[BM25] Found {len(all_scored)} scored chunks across {len(filenames)} file(s)")

    return all_scored[:top_k]


def clear_bm25_index(filename: str):
    """Remove BM25 index for a specific file. Called when file is re-uploaded."""
    if filename in _bm25_indexes:
        del _bm25_indexes[filename]
        _save_to_disk()
        print(f"[BM25] Cleared index for '{filename}'")


def get_indexed_files() -> list:
    """Returns list of all files currently indexed in BM25."""
    return list(_bm25_indexes.keys())