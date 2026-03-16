from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from models_request.request_models import QueryRequest
from rag.rag_pipeline import (
    build_prompt,
    _build_context,
    _build_answer_prompt,
    _is_grounded,
)
from utils.memory import add_to_memory
from utils.query_router import classify_query
from utils.query_rewriter import rewrite_query
from aimodel.llamamodel import generate_response_ollama_stream, generate_response_ollama
from auth.auth_deps import get_current_user

router = APIRouter()

FALLBACK_RESPONSE = (
    "I could not find a reliable answer to your question in the provided documents. "
    "Please try rephrasing your question or check if the relevant information "
    "is in the uploaded document."
)


async def stream_with_grounding(query: str, filenames: list):
    """
    Streaming pipeline with grounding check:

    1. Classify → smalltalk shortcut
    2. Rewrite query if needed
    3. Retrieve context (hybrid search)
    4. Stream answer token by token
    5. After streaming completes, run grounding check
       - If grounded   → already streamed  save to memory
       - If ungrounded → stream a correction message + retry non-streaming
    """

    # ── Smalltalk shortcut ────────────────────────────────────────────────
    query_type = classify_query(query)
    if query_type == "smalltalk":
        prompt = f"User: {query}\nRespond friendly in one short sentence."
        full = ""
        for token in generate_response_ollama_stream(prompt):
            full += token
            yield token
        add_to_memory(query, full)
        return

    # ── Rewrite if needed ─────────────────────────────────────────────────
    rewritten_query = rewrite_query(query)

    # ── ATTEMPT 1: retrieve context ───────────────────────────────────────
    context, parent_docs = _build_context(rewritten_query, filenames, top_k=4)

    if not parent_docs:
        yield "I don't know based on the provided documents."
        return

    prompt = _build_answer_prompt(rewritten_query, context)

    # ── Stream answer ─────────────────────────────────────────────────────
    full_answer = ""
    for token in generate_response_ollama_stream(prompt):
        full_answer += token
        yield token

    # ── Grounding check AFTER streaming ───────────────────────────────────
    if _is_grounded(full_answer, context):
        print(f"[STREAM]  Answer grounded")
        add_to_memory(query, full_answer)
        return

    # ── Not grounded → stream correction + retry ─────────────────────────
    print(f"[STREAM] Answer not grounded — retrying")

    # Stream a separator so user knows we're refining
    correction_msg = "\n\n[Refining answer from document...]\n\n"
    yield correction_msg

    # Attempt 2: retry with original query, more chunks
    context2, parent_docs2 = _build_context(query, filenames, top_k=6)

    if not parent_docs2:
        yield FALLBACK_RESPONSE
        return

    prompt2      = _build_answer_prompt(query, context2)
    full_answer2 = ""

    for token in generate_response_ollama_stream(prompt2):
        full_answer2 += token
        yield token

    # Final grounding check
    if _is_grounded(full_answer2, context2):
        print(f"[STREAM]  Attempt 2 grounded")
        add_to_memory(query, full_answer2)
    else:
        print(f"[STREAM]  Both attempts failed — sending fallback")
        yield f"\n\n{FALLBACK_RESPONSE}"


@router.post("/chat")
async def chat(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    return StreamingResponse(
        stream_with_grounding(request.query, request.filenames),
        media_type="text/plain"
    )