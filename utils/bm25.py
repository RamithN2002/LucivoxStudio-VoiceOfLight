"""
bm25.py — public interface for BM25 search.
Real implementation lives in bm25_store.py.
This file kept for backward compatibility.
"""

from utils.bm25store import bm25_search as _bm25_search


def bm25_search(query: str, filenames: list, top_k: int = 4) -> list:
    """
    Search BM25 index for given filenames.
    Returns list of (chunk_text, score, filename) sorted by score.
    """
    return _bm25_search(query, filenames, top_k)