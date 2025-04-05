# context_store.py
import time
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the embedding model (choose another model if desired)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a new ChromaDB client using the new API (no arguments)
client = chromadb.Client()

# Create (or get) a collection for conversation context.
collection = client.get_or_create_collection(name="conversation_context")

def add_context(session_id: str, message: str, role: str):
    message_id = f"{session_id}_{time.time()}"  # Unique ID using session and timestamp.
    embedding = embedding_model.encode(message).tolist()
    collection.add(
        documents=[message],
        ids=[message_id],
        metadatas=[{"session_id": session_id, "role": role}],
        embeddings=[embedding]
    )

def get_context(session_id: str, query: str, n_results: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"session_id": session_id}
    )
    return results.get("documents", [[]])[0]
