"""
Conversation-Aware Query Rewriter

Two responsibilities:

1. rewrite_query(query)        → for the LLM PROMPT
   Only triggers on pronouns + history. Uses Ollama.

2. build_retrieval_query(query) → for ChromaDB + BM25 SEARCH
   Always injects key terms from history. No LLM needed.
   Makes ALL follow-up questions context-aware in retrieval.
"""

import re
from utils.memory import format_history, get_history


STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","this","that","these","those","it","its","in",
    "on","at","to","for","of","and","or","but","with","from","by","as",
    "into","through","about","i","you","we","they","he","she","what",
    "how","why","when","where","which","who","whom","whose","yes","no",
    "not","also","just","very","so","if","then","than","because","while",
    "although","however","therefore","thus","hence","said","says","get",
    "gets","got","give","gives","given","make","makes","made","take","takes"
}

PRONOUNS = {
    "it","its","they","them","their","theirs",
    "this","that","these","those","he","she",
    "his","her","hers","we","our","ours"
}


def _extract_keywords(text: str, max_keywords: int = 8) -> list:
    """Extracts meaningful content words from text. Removes stop words."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in STOP_WORDS]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    # Longer words = more domain-specific = more valuable for retrieval
    unique.sort(key=len, reverse=True)
    return unique[:max_keywords]


def _has_pronoun(query: str) -> bool:
    """True if query has a pronoun needing resolution."""
    words = set(re.findall(r'\b\w+\b', query.lower()))
    return bool(words & PRONOUNS)


# ── Function 1: rewrite_query → for LLM PROMPT ───────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Rewrites query for the LLM prompt only when pronouns detected.
    Returns original query if no pronouns or no history.
    """
    history = format_history()

    if not history.strip() or not _has_pronoun(query):
        return query

    try:
        import ollama
        prompt = f"""Conversation so far:
{history}

Rewrite this follow-up question as a fully standalone question.
Replace all pronouns with their actual referents from the conversation.
Return only the rewritten question, nothing else.

Question: {query}
Rewritten:"""

        response = ollama.chat(
            model="llama3.1:8b-instruct-q4_K_M",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite questions to be standalone. "
                        "Replace pronouns with what they refer to. "
                        "Return only the rewritten question."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0,
                "num_ctx": 512,
                "num_predict": 80,
                "num_thread": 8,
            }
        )
        rewritten = response["message"]["content"].strip()
        return rewritten if rewritten and len(rewritten) > 5 else query

    except Exception as e:
        print(f"[REWRITER] Ollama failed ({e}), using original")
        return query


# ── Function 2: build_retrieval_query → for ChromaDB + BM25 ──────────────────

def build_retrieval_query(query: str) -> str:
    """
    Builds enriched retrieval query by injecting key terms
    from the last 2 conversation turns.

    Makes ALL follow-up questions context-aware — not just pronoun ones.

    Example:
      History:  "Precision is the ratio of true positives..."
      Query:    "What about in medical diagnosis?"
      Returns:  "precision positives ratio diagnosis What about in medical diagnosis?"

    No LLM needed — pure keyword extraction, ~0ms.
    Returns original query unchanged if no history.
    """
    history_turns = get_history()  # list of (question, answer) tuples

    if not history_turns:
        return query

    # Use last 2 turns only — older context is less relevant
    recent_turns = history_turns[-2:]
    context_text = " ".join([f"{q} {a}" for q, a in recent_turns])

    context_keywords = _extract_keywords(context_text, max_keywords=6)

    if not context_keywords:
        return query

    # Keywords first → boosts their weight in BM25 scoring
    enriched = f"{' '.join(context_keywords)} {query}"

    print(f"[RETRIEVAL QUERY] Original: '{query}'")
    print(f"[RETRIEVAL QUERY] Enriched: '{enriched}'")

    return enriched