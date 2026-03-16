"""
Semantic Chunker

Splits documents at topic boundaries instead of character count.

Algorithm:
  1. Split text into sentences using regex
  2. Embed each sentence using existing embedding model
  3. Compute cosine similarity between consecutive sentences
  4. Find "breakpoints" where similarity drops sharply (topic change)
  5. Group sentences between breakpoints into chunks
  6. Apply min/max size guards

Why this is better than RecursiveCharacterTextSplitter:
  - Never cuts mid-sentence
  - Never cuts mid-paragraph
  - Each chunk is a complete thought
  - LLM gets cleaner, more coherent context

Uses your existing SentenceTransformer model — no new dependencies.
"""

import re
import numpy as np
from embeddings.embedding import generate_embeddings


# ── Config ────────────────────────────────────────────────────────────────────
MIN_CHUNK_SIZE  = 200   # chars — merge chunks smaller than this
MAX_CHUNK_SIZE  = 1200  # chars — split chunks larger than this
BREAKPOINT_THRESHOLD = 0.35  # similarity drop below this = topic change
# Lower = more sensitive (more chunks), Higher = less sensitive (fewer chunks)
# 0.35 is a good default for technical documents


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list:
    """
    Splits text into sentences using regex.
    Handles common abbreviations (e.g., Dr., Fig., vs.) to avoid false splits.
    """
    # Protect common abbreviations from being split
    abbreviations = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Fig.", "vs.",
        "e.g.", "i.e.", "et al.", "etc.", "No.", "Vol.", "pp."
    ]
    protected = text
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on sentence endings: . ! ? followed by space + capital letter
    # Also split on double newlines (paragraph breaks)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?:\n\n+)', protected)

    # Restore abbreviations
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

    # Filter out empty strings and very short fragments
    sentences = [s for s in sentences if len(s) > 20]

    return sentences


# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Find breakpoints ──────────────────────────────────────────────────────────

def _find_breakpoints(embeddings: list, threshold: float) -> list:
    """
    Finds indices where topic changes occur.

    Computes similarity between consecutive sentence embeddings.
    When similarity drops below threshold → breakpoint (topic changed).

    Also uses a "valley detection" approach:
    If similarity at index i is significantly lower than both neighbors,
    it's a strong breakpoint even if above threshold.

    Returns list of indices AFTER which a new chunk should start.
    """
    breakpoints = []
    similarities = []

    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(
            np.array(embeddings[i]),
            np.array(embeddings[i + 1])
        )
        similarities.append(sim)

    if not similarities:
        return []

    # Method 1: Hard threshold — similarity below threshold = breakpoint
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i)

    # Method 2: Valley detection — local minimum significantly below neighbors
    for i in range(1, len(similarities) - 1):
        prev_sim = similarities[i - 1]
        curr_sim = similarities[i]
        next_sim = similarities[i + 1]

        # If current similarity is much lower than both neighbors
        if curr_sim < (prev_sim - 0.15) and curr_sim < (next_sim - 0.15):
            if i not in breakpoints:
                breakpoints.append(i)

    breakpoints.sort()
    return breakpoints


# ── Group sentences into chunks ───────────────────────────────────────────────

def _group_sentences(sentences: list, breakpoints: list) -> list:
    """
    Groups sentences into chunks based on breakpoints.
    Applies min/max size guards.
    """
    if not sentences:
        return []

    # Split sentences into groups at breakpoints
    groups = []
    start = 0

    for bp in breakpoints:
        group = sentences[start:bp + 1]
        if group:
            groups.append(group)
        start = bp + 1

    # Add remaining sentences
    if start < len(sentences):
        groups.append(sentences[start:])

    # Convert groups to text chunks
    chunks = [" ".join(group).strip() for group in groups if group]

    # ── Apply size guards ─────────────────────────────────────────────────

    final_chunks = []
    buffer = ""

    for chunk in chunks:

        # Too small → merge with buffer
        if len(chunk) < MIN_CHUNK_SIZE:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
            continue

        # Flush buffer if it's big enough
        if buffer:
            if len(buffer) >= MIN_CHUNK_SIZE:
                final_chunks.append(buffer)
            else:
                # Buffer still too small — prepend to current chunk
                chunk = (buffer + " " + chunk).strip()
            buffer = ""

        # Too large → split in half at sentence boundary
        if len(chunk) > MAX_CHUNK_SIZE:
            mid = len(chunk) // 2
            # Find nearest sentence boundary to midpoint
            split_point = chunk.rfind(". ", 0, mid)
            if split_point == -1:
                split_point = mid

            part1 = chunk[:split_point + 1].strip()
            part2 = chunk[split_point + 1:].strip()

            if part1:
                final_chunks.append(part1)
            if part2:
                final_chunks.append(part2)
        else:
            final_chunks.append(chunk)

    # Flush remaining buffer
    if buffer:
        if final_chunks:
            # Merge with last chunk
            final_chunks[-1] = (final_chunks[-1] + " " + buffer).strip()
        else:
            final_chunks.append(buffer)

    return [c for c in final_chunks if c.strip()]


# ── Main function ─────────────────────────────────────────────────────────────

def semantic_chunk(text: str, threshold: float = BREAKPOINT_THRESHOLD) -> list:
    """
    Splits a document into semantically coherent chunks.

    Args:
        text:      full document text
        threshold: similarity threshold for topic change detection
                   lower = more chunks, higher = fewer chunks

    Returns:
        list of chunk strings, each representing a complete thought
    """
    print(f"[SEMANTIC CHUNKER] Starting chunking (threshold={threshold})")

    # Step 1: Split into sentences
    sentences = _split_sentences(text)
    print(f"[SEMANTIC CHUNKER] {len(sentences)} sentences found")

    if len(sentences) <= 2:
        # Too few sentences — return as single chunk
        return [text.strip()]

    # Step 2: Embed all sentences in one batch call (efficient)
    print(f"[SEMANTIC CHUNKER] Embedding {len(sentences)} sentences...")
    embeddings = generate_embeddings(sentences)
    print(f"[SEMANTIC CHUNKER] Embeddings done")

    # Step 3: Find breakpoints
    breakpoints = _find_breakpoints(embeddings, threshold)
    print(f"[SEMANTIC CHUNKER] Found {len(breakpoints)} topic breakpoints")

    # Step 4: Group into chunks with size guards
    chunks = _group_sentences(sentences, breakpoints)
    print(f"[SEMANTIC CHUNKER] Final chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"[SEMANTIC CHUNKER] Chunk {i+1}: {len(chunk)} chars — '{chunk[:60]}...'")

    return chunks