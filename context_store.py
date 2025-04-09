import time
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize the embedding model.
# This model converts text into numerical embeddings that can be stored and queried.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a new ChromaDB client.
# The client here is used to interact with the vector database.
client = chromadb.Client()

# Create (or get) a collection for conversation context.
# A collection in ChromaDB stores documents along with their embeddings and metadata.
collection = client.get_or_create_collection(name="conversation_context")

def add_context(session_id: str, message: str, role: str):
    """
    Store a conversation message in the vector database.
    
    This function takes the session ID, the message text, and the role of the sender
    (e.g., "user" or "assistant"). It generates a unique ID for the message using the
    session ID and the current timestamp, converts the message into an embedding using
    the SentenceTransformer model, and then adds the message along with its metadata
    to the ChromaDB collection.
    
    Args:
        session_id (str): Unique identifier for the conversation session.
        message (str): The text message to be stored.
        role (str): The role of the sender (e.g., "user" or "assistant").
    """
    # Create a unique identifier using the session ID and current timestamp.
    message_id = f"{session_id}_{time.time()}"
    # Encode the message into a numerical embedding.
    embedding = embedding_model.encode(message).tolist()
    # Add the message and its metadata to the collection.
    collection.add(
        documents=[message],
        ids=[message_id],
        metadatas=[{"session_id": session_id, "role": role}],
        embeddings=[embedding]
    )

def get_context(session_id: str, query: str, n_results: int = 3):
    """
    Retrieve the most relevant conversation context for a given query.
    
    This function converts the provided query into an embedding, then queries the 
    ChromaDB collection to retrieve the top 'n_results' documents that belong to the 
    specified session and that are most similar to the query.
    
    Args:
        session_id (str): Unique identifier for the conversation session.
        query (str): The text query for which to retrieve contextual messages.
        n_results (int, optional): Number of similar messages to retrieve. Defaults to 3.
        
    Returns:
        list: A list of documents (message texts) that match the query. 
              If no documents are found, returns an empty list.
    """
    # Convert the query text into an embedding.
    query_embedding = embedding_model.encode(query).tolist()
    # Query the collection for the top 'n_results' similar documents within the same session.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"session_id": session_id}
    )
    # Return the list of retrieved documents; if none found, return an empty list.
    return results.get("documents", [[]])[0]
