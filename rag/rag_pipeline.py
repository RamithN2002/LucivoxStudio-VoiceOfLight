from utils.query_router import classify_query
from utils.query_rewriter import rewrite_query, build_retrieval_query
from utils.memory import format_history, add_to_memory
from utils.rerank import rerank
from utils.context_compression import compress_context
from rag.parent_retrieval import retrieve_parent_documents


# ── Answer Grounding Check ────────────────────────────────────────────────────

def _is_grounded(answer: str, context: str) -> bool:
    """
    Verifies whether the answer is grounded in the retrieved context.
    Returns True if grounded, False if hallucinated.
    Falls back to True if Ollama fails.
    """
    import ollama

    prompt = f"""Context:
        {context}

        Answer:
        {answer}

        Is every single claim in the Answer directly supported by the Context above?
        Reply with only one word: YES or NO."""

    try:
        response = ollama.chat(
            model="llama3.1:8b-instruct-q4_K_M",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict fact checker. "
                        "Reply only YES or NO. "
                        "If even one claim is not in the context, reply NO."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0,
                "num_ctx": 1024,
                "num_predict": 5,
                "num_thread": 8,
            }
        )
        result = response["message"]["content"].strip().upper()
        is_grounded = result.startswith("YES")
        print(f"[GROUNDING] Result: {result} → grounded={is_grounded}")
        return is_grounded

    except Exception as e:
        print(f"[GROUNDING] Check failed ({e}), assuming grounded")
        return True


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(retrieval_query: str, filenames: list, top_k: int = 4) -> tuple:
    """
    Retrieves, reranks and compresses chunks for a retrieval query.
    Returns (context_string, parent_docs).
    """
    parent_docs = retrieve_parent_documents(retrieval_query, filenames, top_k=top_k)

    if not parent_docs:
        return None, []

    top_docs = rerank(retrieval_query, parent_docs, top_k=3) if len(parent_docs) > 3 else parent_docs
    compressed_docs = compress_context(retrieval_query, top_docs)
    context = "\n---\n".join(compressed_docs[:3])

    return context, parent_docs


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_answer_prompt(query: str, context: str) -> str:
    """
    Builds the final LLM prompt.
    Uses the REWRITTEN query (clean natural language) not the enriched retrieval query.
    """
    history = format_history()
    history_section = f"History:\n{history}\n" if history.strip() else ""

    return f"""{history_section}Context:
{context}

Q: {query}
A (use context only, be concise, say "I don't know" if not found):"""


# ── build_prompt (for streaming in chat_routes) ───────────────────────────────

def build_prompt(query: str, filenames: list):
    """
    Builds prompt for streaming.
    Returns (prompt, query_type, rewritten_query).

    Uses:
      - build_retrieval_query → enriched query for ChromaDB + BM25
      - rewrite_query         → clean query for LLM prompt
    """
    query_type = classify_query(query)

    if query_type == "smalltalk":
        prompt = f"User: {query}\nRespond friendly in one short sentence."
        return prompt, "smalltalk", query

    # Two separate queries:
    # retrieval_query → context-enriched, used for ChromaDB + BM25
    # prompt_query    → clean rewritten, used in LLM prompt
    retrieval_query = build_retrieval_query(query)
    prompt_query    = rewrite_query(query)

    context, parent_docs = _build_context(retrieval_query, filenames, top_k=4)

    if not parent_docs:
        return None, "no_docs", prompt_query

    prompt = _build_answer_prompt(prompt_query, context)
    return prompt, "knowledge", prompt_query


# ── rag_query (non-streaming with full grounding check) ───────────────────────

def rag_query(query: str, filenames: list) -> str:
    """
    Full RAG pipeline with grounding check.

    Attempt 1: enriched retrieval query → generate → ground check
    Attempt 2: original query, more chunks → generate → ground check
    Fallback:  honest "could not find reliable answer"
    """
    from aimodel.llamamodel import generate_response_ollama

    query_type = classify_query(query)

    if query_type == "smalltalk":
        prompt = f"User: {query}\nRespond friendly in one short sentence."
        answer = generate_response_ollama(prompt)
        add_to_memory(query, answer)
        return answer

    # ── Two separate queries ──────────────────────────────────────────────
    retrieval_query = build_retrieval_query(query)  # for ChromaDB + BM25
    prompt_query    = rewrite_query(query)          # for LLM prompt

    # ── Attempt 1 ─────────────────────────────────────────────────────────
    print(f"[PIPELINE] Attempt 1 | retrieval='{retrieval_query}'")

    context, parent_docs = _build_context(retrieval_query, filenames, top_k=4)

    if not parent_docs:
        return "I don't know based on the provided documents."

    prompt = _build_answer_prompt(prompt_query, context)
    answer = generate_response_ollama(prompt)

    if _is_grounded(answer, context):
        print(f"[PIPELINE]  Attempt 1 grounded")
        add_to_memory(query, answer)
        return answer

    # ── Attempt 2: retry with original query, more chunks ────────────────
    print(f"[PIPELINE] Not grounded — retrying with original query")

    context2, parent_docs2 = _build_context(query, filenames, top_k=6)

    if not parent_docs2:
        return "I could not find a reliable answer in the provided documents."

    prompt2 = _build_answer_prompt(query, context2)
    answer2 = generate_response_ollama(prompt2)

    if _is_grounded(answer2, context2):
        print(f"[PIPELINE] Attempt 2 grounded")
        add_to_memory(query, answer2)
        return answer2

    print(f"[PIPELINE]  Both attempts failed grounding")
    return (
        "I could not find a reliable answer to your question in the provided documents. "
        "Please try rephrasing your question or check if the relevant information "
        "is in the uploaded document."
    )