import ollama


def generate_response_ollama(prompt: str) -> str:
    """
    Tuned for 8-core CPU + 32GB RAM.
    - llama3.1:8b-instruct-q4_K_M fits well in 32GB
    - num_thread=8 uses all cores
    - num_ctx=1536 is enough for RAG (context + question)
    - num_predict=200 forces concise answers = faster
    - temperature=0 = no sampling overhead
    """
    response = ollama.chat(
        model="llama3.1:8b-instruct-q4_K_M",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. "
                    "Answer using ONLY the provided context. "
                    "Be concise — max 3 sentences. "
                    "If the answer is not in the context, say exactly: I don't know."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 1536,
            "num_predict": 200,
            "num_thread": 8,
            "repeat_penalty": 1.1,
            "top_k": 10,         
            "top_p": 0.9,
        }
    )
    return response["message"]["content"]


def generate_response_ollama_stream(prompt: str):
    """
    ✅ Streaming version — yields tokens as they are generated
    so the user sees words appearing immediately instead of waiting
    for the full response.
    """
    stream = ollama.chat(
        model="llama3.1:8b-instruct-q4_K_M",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise assistant. "
                    "Answer using ONLY the provided context. "
                    "Be concise — max 3 sentences. "
                    "If the answer is not in the context, say exactly: I don't know."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": 1536,
            "num_predict": 200,
            "num_thread": 8,
            "repeat_penalty": 1.1,
            "top_k": 10,
            "top_p": 0.9,
        },
        stream=True  
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token