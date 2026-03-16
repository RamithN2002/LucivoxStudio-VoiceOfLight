from sentence_transformers import SentenceTransformer
from config.setting import EMBEDDING_MODEL
from functools import lru_cache
import numpy as np

model = SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(documents):
    return model.encode(documents)

@lru_cache(maxsize=256)
def get_query_embedding(query: str):
    return model.encode([query])[0]