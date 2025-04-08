from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent
from context_store import get_context
from langchain_memory import (
    add_message_to_memory, 
    get_formatted_history, 
    get_prompt_template
)
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # default session if none provided

# Load the embedding model globally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define domain prototypes
# The Concordia prototype is very specific: it emphasizes admission to the Concordia Computer Science program.
concordia_prototype = ("Admission to the Concordia University Computer Science program, "
                       "including application requirements, cut-off averages, and process for admission")
ai_prototype = "Artificial Intelligence, machine learning, deep learning, AI research, modern AI breakthroughs"
general_prototype = "General inquiries, everyday questions, general information"

# Pre-compute embeddings for each prototype
concordia_embedding = embedding_model.encode(concordia_prototype, convert_to_tensor=True)
ai_embedding = embedding_model.encode(ai_prototype, convert_to_tensor=True)
general_embedding = embedding_model.encode(general_prototype, convert_to_tensor=True)

def clean_user_message(message: str):
    """Extract just the user's message from potentially complex prompts"""
    if "user:" in message.lower():
        parts = message.split("User:", 1)
        if len(parts) > 1:
            message = parts[1].split("Assistant:", 1)[0].strip()
    return message

def choose_agent(message: str):
    # Clean the message first
    message = clean_user_message(message)
    
    # Compute the embedding for the user message.
    query_embedding = embedding_model.encode(message, convert_to_tensor=True)
    
    # Compute cosine similarities.
    concordia_sim = util.pytorch_cos_sim(query_embedding, concordia_embedding).item()
    ai_sim = util.pytorch_cos_sim(query_embedding, ai_embedding).item()
    general_sim = util.pytorch_cos_sim(query_embedding, general_embedding).item()
    
    # Apply weighting: boost the general similarity by 50%
    general_weight = 1.5
    weighted_general_sim = general_sim * general_weight
    
    # Debug output
    print(f"Similarity Scores => Concordia: {concordia_sim:.3f}, AI: {ai_sim:.3f}, General (weighted): {weighted_general_sim:.3f}")
    
    # Define a minimum threshold. If the best similarity is below this, default to general.
    minimum_threshold = 0.4
    
    scores = {
        "concordia": concordia_sim,
        "ai": ai_sim,
        "general": weighted_general_sim
    }
    
    best_agent = max(scores, key=scores.get)
    
    # If the highest score is below the threshold, default to general.
    if scores[best_agent] < minimum_threshold:
        best_agent = "general"
    
    # Return agent type and function
    if best_agent == "concordia":
        return "concordia", concordia_agent.concordia_agent
    elif best_agent == "ai":
        return "ai", ai_agent.ai_agent
    else:
        return "general", general_agent.general_agent

@app.post("/chat")
def chat(request: ChatRequest):
    # Clean user message first
    clean_message = clean_user_message(request.message)
    
    # Choose the agent before getting context
    agent_type, agent_func = choose_agent(clean_message)
    
    # Get agent response directly with the clean message
    response = agent_func(clean_message)
    
    # IMPORTANT: Only store the clean user message and final response
    # This ensures we don't store prompt templates in the context
    add_message_to_memory(request.session_id, clean_message, role="user")
    add_message_to_memory(request.session_id, response, role="assistant")
    
    return {"message": response}