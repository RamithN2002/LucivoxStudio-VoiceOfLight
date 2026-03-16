import chromadb
from config.setting import CHROMA_DB_PATH

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_or_create_collection(
    name="rag_documents30"
)