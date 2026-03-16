import os
import re
import ollama


def expand_query(query: str) -> list:
    prompt = f"""Generate 4 different search queries to find information about the following question in a document.
Each query should be a different way to search for the same information.
Output ONLY the 4 queries, one per line, no numbering, no explanation, no extra text.

Original question: {query}

4 search queries:"""

    try:
        response = ollama.chat(
            model="llama3.1:8b-instruct-q4_K_M",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query generator. "
                        "Output only the requested queries, one per line. "
                        "No numbering, no bullets, no explanation."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.4,   
                "num_ctx": 512,      
                "num_predict": 100,   
            }
        )

        raw = response["message"]["content"].strip()

        lines = raw.split("\n")
        expanded = []
        for line in lines:
            clean = re.sub(r'^[\d\.\-\*\•\s]+', '', line).strip()
            if clean and len(clean) > 5:
                expanded.append(clean)

        result = [query]
        for q in expanded:
            if q.lower() != query.lower() and q not in result:
                result.append(q)

        result = result[:5]

        print(f"[QUERY EXPANSION] Generated {len(result)} queries: {result}")
        return result

    except Exception as e:
        print(f"[QUERY EXPANSION] LLM expansion failed ({e}), using original query only")
        return [query]