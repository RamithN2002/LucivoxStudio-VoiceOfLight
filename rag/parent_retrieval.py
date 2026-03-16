"""
Hybrid Retrieval — Vector + BM25 + Query Expansion + RRF Fusion

Pipeline:
  1. Expand query into 3-5 variations
  2. For each variation:
       a. Vector search  (ChromaDB semantic)
       b. BM25 search    (keyword exact match)
  3. Merge all results using Reciprocal Rank Fusion (RRF)
  4. Deduplicate by parent_text
  5. Return top_k unique parent docs

Why RRF?
  Simple score averaging fails when vector and BM25 scores are on
  different scales. RRF uses rank positions instead of raw scores,
  so both signals contribute equally regardless of magnitude.
  RRF score = sum(1 / (k + rank_i))  where k=60 (standard constant)
"""

from embeddings.embedding import get_query_embedding
from chroma.chromadb_create_collection import collection
from utils.query_expansion import expand_query
from utils.bm25store import bm25_search

RRF_K = 60  


def _vector_search(query: str, filenames: list, n_results: int) -> list:
    """Single vector search. Returns list of dicts {text, source, score}."""
    if not filenames:
        return []

    query_embedding = get_query_embedding(query)

    where_filter = (
        {"source": filenames[0]}
        if len(filenames) == 1
        else {"source": {"$in": filenames}}
    )

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,
        )
    except Exception as e:
        print(f"[VECTOR] Search error: {e}")
        return []

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    docs = []
    for m, dist in zip(metadatas, distances):
        if m and "parent_text" in m:
            docs.append({
                "text":   m["parent_text"],
                "source": m.get("source", "unknown"),
                "score":  1 - dist,
            })
    return docs


def _rrf_fusion(ranked_lists: list, top_k: int) -> list:
    """
    Reciprocal Rank Fusion across multiple ranked lists.
    Returns list of "[Source: x]\ntext" strings sorted by RRF score.
    """
    rrf_scores:  dict = {}
    text_to_meta: dict = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            text = item["text"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (RRF_K + rank + 1)
            if text not in text_to_meta:
                text_to_meta[text] = item.get("source", "unknown")

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        f"[Source: {text_to_meta[text]}]\n{text}"
        for text, _ in sorted_items[:top_k]
    ]


def retrieve_parent_documents(query: str, filenames: list, top_k: int = 4) -> list:
    """
    Full hybrid retrieval pipeline:
      query expansion → vector search + BM25 → RRF fusion → top_k results
    """
    print(f"[RETRIEVAL] query='{query}' | filenames={filenames}")

    if not filenames:
        return []

    expanded_queries = expand_query(query)
    print(f"[RETRIEVAL] expanded to {len(expanded_queries)} variants")

    fetch_k = top_k * 2  
    all_ranked_lists = []

    for q in expanded_queries:

        vector_results = _vector_search(q, filenames, n_results=fetch_k)
        if vector_results:
            all_ranked_lists.append(vector_results)

        bm25_raw = bm25_search(q, filenames, top_k=fetch_k)
        if bm25_raw:
            all_ranked_lists.append([
                {"text": text, "source": src, "score": score}
                for text, score, src in bm25_raw
            ])

    if not all_ranked_lists:
        print(f"[RETRIEVAL] No results from any search method")
        return []

    print(f"[RETRIEVAL] Fusing {len(all_ranked_lists)} ranked lists via RRF")

    fused = _rrf_fusion(all_ranked_lists, top_k=top_k)

    print(f"[RETRIEVAL] Final chunks after fusion: {len(fused)}")
    return fused