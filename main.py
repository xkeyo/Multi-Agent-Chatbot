from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent
from context_store import add_context, get_context
from sentence_transformers import SentenceTransformer, util
from langchain_memory import LangChainMemoryManager, PromptEngineering
from typing import Dict
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = None  # Make session_id optional

class ChatResponse(BaseModel):
    message: str
    session_id: str  # Always return the session ID

# Load the embedding model globally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define domain prototypes
concordia_prototype = ("Admission to the Concordia University Computer Science program, "
                       "including application requirements, cut-off averages, and process for admission")
ai_prototype = "Artificial Intelligence, machine learning, deep learning, AI research, modern AI breakthroughs"
general_prototype = "General inquiries, everyday questions, general information"

# Pre-compute embeddings for each prototype
concordia_embedding = embedding_model.encode(concordia_prototype, convert_to_tensor=True)
ai_embedding = embedding_model.encode(ai_prototype, convert_to_tensor=True)
general_embedding = embedding_model.encode(general_prototype, convert_to_tensor=True)

# Store active memory managers
memory_managers: Dict[str, LangChainMemoryManager] = {}

def get_memory_manager(session_id: str) -> LangChainMemoryManager:
    """Get or create a memory manager for the given session"""
    if session_id not in memory_managers:
        memory_managers[session_id] = LangChainMemoryManager(session_id=session_id, memory_type="window", k=5)
    return memory_managers[session_id]

def choose_agent_and_domain(message: str):
    # Compute the embedding for the user message
    query_embedding = embedding_model.encode(message, convert_to_tensor=True)
    
    # Compute cosine similarities
    concordia_sim = util.pytorch_cos_sim(query_embedding, concordia_embedding).item()
    ai_sim = util.pytorch_cos_sim(query_embedding, ai_embedding).item()
    general_sim = util.pytorch_cos_sim(query_embedding, general_embedding).item()
    
    # Apply weighting: boost the general similarity by 50%
    general_weight = 1.5
    weighted_general_sim = general_sim * general_weight
    
    # Debug output
    print(f"Similarity Scores => Concordia: {concordia_sim:.3f}, AI: {ai_sim:.3f}, General (weighted): {weighted_general_sim:.3f}")
    
    # Define a minimum threshold. If the best similarity is below this, default to general
    minimum_threshold = 0.4
    
    scores = {
        "concordia": concordia_sim,
        "ai": ai_sim,
        "general": weighted_general_sim
    }
    
    best_domain = max(scores, key=scores.get)
    
    # If the highest score is below the threshold, default to general
    if scores[best_domain] < minimum_threshold:
        best_domain = "general"
    
    if best_domain == "concordia":
        return concordia_agent.concordia_agent, best_domain
    elif best_domain == "ai":
        return ai_agent.ai_agent, best_domain
    else:
        return general_agent.general_agent, best_domain

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Generate a session ID if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
    
    # Get the memory manager for this session
    memory_manager = get_memory_manager(request.session_id)
    
    # Retrieve relevant context from ChromaDB for the current session
    previous_contexts = get_context(request.session_id, request.message, n_results=3)
    context_text = "\n".join(previous_contexts) if previous_contexts else ""
    
    # Choose the agent based on the message content
    agent_func, domain = choose_agent_and_domain(request.message)
    
    # Create a domain-specific prompt using the PromptEngineering class
    custom_prompt = PromptEngineering.create_domain_specific_prompt(
        domain=domain,
        user_input=request.message,
        context=context_text
    )
    
    # Add the user message to memory
    memory_manager.add_user_message(request.message)
    
    response = memory_manager.generate_response(request.message, custom_prompt=custom_prompt)

    
    # Add the assistant's response to memory
    memory_manager.add_ai_message(response)
    
    # Save the current user message and assistant response in ChromaDB
    add_context(request.session_id, request.message, role="user")
    add_context(request.session_id, response, role="assistant")
    
    return {"message": response, "session_id": request.session_id}