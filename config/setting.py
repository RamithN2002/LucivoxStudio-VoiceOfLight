import os 
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
MODEL_NAME_2 = os.getenv("MODEL_NAME_2")