def compress_context(query: str, documents: list) -> list:

    query_keywords = set(
        word for word in query.lower().split()
        if len(word) > 3
    )

    compressed_docs = []

    for doc in documents:
        doc_lower = doc.lower()
        matched = any(keyword in doc_lower for keyword in query_keywords)

        if matched:
            compressed_docs.append(doc)
        else:
            compressed_docs.append(doc[:500])

    return compressed_docs