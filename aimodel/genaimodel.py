from google import genai
from config.setting import GOOGLE_API_KEY, MODEL_NAME
from functools import lru_cache

client = genai.Client(api_key=GOOGLE_API_KEY)
genai_model = MODEL_NAME


def generate_response(prompt: str) -> str:
    return _cached_generate(prompt)

@lru_cache(maxsize=128)
def _cached_generate(prompt: str) -> str:
    response = client.models.generate_content(
        model=genai_model,
        contents=prompt
    )
    return response.text