"""
Selective Reranker

Two-stage reranking strategy:

Stage 1 — Cosine Similarity (always runs, fast ~5ms)
  Uses cached query embedding vs chunk embeddings.
  Produces initial ranked list with confidence scores.

Stage 2 — Cross Encoder (selective, ~1-2s on CPU)
  Only runs when top chunks have similar cosine scores (ambiguous ranking).
  Reads query + chunk TOGETHER for deeper relevance understanding.
  Threshold: if gap between top 2 scores < CONFIDENCE_THRESHOLD → run Cross Encoder

Why this works:
  When retrieval is confident (clear score gap) → cosine is enough.
  When retrieval is uncertain (similar scores)  → Cross Encoder breaks the tie.
  Result: best accuracy at minimum latency cost.

Cross Encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Small and fast (6 layers)
  - Trained specifically for passage relevance scoring
  - Works well on CPU
"""

import numpy as np
from embeddings.embedding import get_query_embedding, generate_embeddings

# Threshold — if gap between top 2 cosine scores < this → run Cross Encoder
# 0.15 means: if two chunks score within 15% of each other → ambiguous → rerank
CONFIDENCE_THRESHOLD = 0.15

# Cross Encoder loaded lazily (only when needed)
_cross_encoder = None


def _load_cross_encoder():
    """Lazy load Cross Encoder — only imported when first needed."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[RERANK] Cross Encoder loaded")
        except Exception as e:
            print(f"[RERANK] Could not load Cross Encoder: {e}")
            _cross_encoder = None
    return _cross_encoder


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _cosine_rerank(query: str, documents: list) -> list:
    """
    Stage 1: Rank documents by cosine similarity with query.
    Returns list of (doc, score) tuples sorted descending.
    Uses cached query embedding — no re-computation.
    """
    if not documents:
        return []

    query_emb   = get_query_embedding(query)
    doc_embeddings = generate_embeddings(documents)

    scored = [
        (doc, _cosine_similarity(query_emb, doc_emb))
        for doc, doc_emb in zip(documents, doc_embeddings)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _cross_encoder_rerank(query: str, documents: list) -> list:
    """
    Stage 2: Rerank documents using Cross Encoder.
    Reads query + document TOGETHER for deeper relevance scoring.
    Returns list of (doc, score) tuples sorted descending.
    """
    ce = _load_cross_encoder()

    if ce is None:
        # Cross Encoder unavailable — fall back to cosine ranking
        print("[RERANK] Cross Encoder unavailable, using cosine fallback")
        return [(doc, 0.0) for doc in documents]

    pairs  = [[query, doc] for doc in documents]
    scores = ce.predict(pairs)

    scored = list(zip(documents, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


def _is_ambiguous(cosine_scored: list) -> bool:
    """
    Returns True if the top chunks have similar cosine scores.
    Ambiguous = gap between top 2 scores < CONFIDENCE_THRESHOLD.
    This triggers Cross Encoder reranking.
    """
    if len(cosine_scored) < 2:
        return False

    top_score    = cosine_scored[0][1]
    second_score = cosine_scored[1][1]
    gap          = top_score - second_score

    is_amb = gap < CONFIDENCE_THRESHOLD
    print(f"[RERANK] Top scores: {top_score:.3f}, {second_score:.3f} | "
          f"gap={gap:.3f} | ambiguous={is_amb}")
    return is_amb


def rerank(query: str, documents: list, top_k: int = 3) -> list:
    """
    Selective two-stage reranker.

    Stage 1: Cosine similarity ranking (always, fast)
    Stage 2: Cross Encoder (only when top scores are too similar)

    Args:
        query:     search query
        documents: list of document strings to rerank
        top_k:     number of top documents to return

    Returns:
        list of top_k document strings, best first
    """
    if not documents:
        return []

    if len(documents) == 1:
        return documents

    # ── Stage 1: Cosine similarity ────────────────────────────────────────
    cosine_scored = _cosine_rerank(query, documents)

    print(f"[RERANK] Stage 1 cosine scores: "
          f"{[round(s, 3) for _, s in cosine_scored]}")

    # ── Decision: is ranking ambiguous? ──────────────────────────────────
    if not _is_ambiguous(cosine_scored):
        # Confident ranking — cosine result is good enough
        print(f"[RERANK] Confident ranking — skipping Cross Encoder (fast path)")
        return [doc for doc, _ in cosine_scored[:top_k]]

    # ── Stage 2: Cross Encoder (only top candidates) ──────────────────────
    # Only rerank the top min(6, len) candidates — no need to run CE on all
    candidates = [doc for doc, _ in cosine_scored[:min(6, len(cosine_scored))]]

    print(f"[RERANK] Ambiguous — running Cross Encoder on {len(candidates)} candidates")

    ce_scored = _cross_encoder_rerank(query, candidates)

    print(f"[RERANK] Stage 2 Cross Encoder scores: "
          f"{[round(float(s), 3) for _, s in ce_scored]}")

    return [doc for doc, _ in ce_scored[:top_k]]