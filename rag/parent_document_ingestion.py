import uuid
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.embedding import generate_embeddings
from chroma.chromadb_create_collection import collection
from rag.parent_store import add_parent
from utils.bm25store import add_to_bm25_index, clear_bm25_index
from utils.semantic_chunker import semantic_chunk


# ── Child splitter (still character-based for small child chunks) ─────────────
# Child chunks are small (200 chars) and used only for vector search embedding
# Character splitting is fine at this small scale
# Semantic splitting is only needed for parent chunks (the ones shown to LLM)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)


# ── Topic extractor ───────────────────────────────────────────────────────────

def _extract_topic(chunk: str, max_len: int = 60) -> str:
    """
    Extracts a short topic label from the first meaningful sentence of a chunk.
    No LLM needed — fast and free.
    """
    first_sentence = re.split(r'[.\n]', chunk.strip())[0][:120]

    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","this","that","these",
        "those","it","its","in","on","at","to","for","of","and",
        "or","but","with","from","by","as","into","through","about"
    }

    words = first_sentence.lower().split()
    topic_words = [w for w in words if w not in stopwords and len(w) > 2]
    topic = " ".join(topic_words).strip().capitalize()[:max_len]

    return topic if topic else "General content"


def _build_chunk_header(filename: str, chunk_index: int, topic: str) -> str:
    """Builds context header: [Source: XAI.pdf | Chunk: 3 | Topic: ...]"""
    display_name = filename.split("__")[-1] if "__" in filename else filename
    return f"[Source: {display_name} | Chunk: {chunk_index + 1} | Topic: {topic}]"


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_document(text: str, filename: str):
    """
    Ingests a document into ChromaDB and BM25 index.

    Changes from previous version:
      - Uses semantic_chunk() instead of RecursiveCharacterTextSplitter
        for parent chunks — splits at topic boundaries, not character count
      - Child chunks still use character splitting (fine at 200 char scale)
      - Context headers still added to every parent chunk
    """

    # Clear old BM25 index for re-uploads
    clear_bm25_index(filename)

    # ── Step 1: Semantic chunking for parent chunks ───────────────────────────
    print(f"[INGEST] Starting semantic chunking for '{filename}'")
    parent_chunks = semantic_chunk(text)
    print(f"[INGEST] '{filename}': {len(parent_chunks)} semantic parent chunks")

    all_enriched_chunks = []

    for chunk_index, parent_chunk in enumerate(parent_chunks):

        parent_id = str(uuid.uuid4())

        # ── Build contextual header ───────────────────────────────────────────
        topic  = _extract_topic(parent_chunk)
        header = _build_chunk_header(filename, chunk_index, topic)

        # Enriched chunk = header + original text
        enriched_chunk = f"{header}\n{parent_chunk}"

        # Store enriched parent in memory store
        add_parent(parent_id, enriched_chunk)

        # Collect for BM25
        all_enriched_chunks.append(enriched_chunk)

        # ── Child chunks for vector search ────────────────────────────────────
        # Split original chunk (not enriched) — headers confuse embeddings
        child_chunks = child_splitter.split_text(parent_chunk)

        if not child_chunks:
            # Very short chunk — use as single child
            child_chunks = [parent_chunk]

        embeddings = generate_embeddings(child_chunks)

        collection.add(
            documents=child_chunks,
            embeddings=embeddings,
            metadatas=[{
                "parent_id":   parent_id,
                "parent_text": enriched_chunk,  # enriched version for LLM
                "source":      filename,
                "topic":       topic,
                "chunk_index": chunk_index,
            }] * len(child_chunks),
            ids=[f"{parent_id}_{i}" for i in range(len(child_chunks))]
        )

        print(f"[INGEST] Chunk {chunk_index + 1}/{len(parent_chunks)}: "
              f"topic='{topic}' | {len(child_chunks)} child chunks")

    # ── Build BM25 index ──────────────────────────────────────────────────────
    add_to_bm25_index(filename, all_enriched_chunks)

    print(f"[INGEST] ✅ '{filename}' complete: "
          f"{len(parent_chunks)} semantic chunks ingested")