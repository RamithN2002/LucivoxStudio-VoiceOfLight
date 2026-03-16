def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):

    scores = {}

    for rank, doc in enumerate(vector_docs):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    ranked_docs = sorted(scores, key=scores.get, reverse=True)

    return ranked_docs