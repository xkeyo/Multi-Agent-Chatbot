from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent
from context_store import add_context, get_context
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

def choose_agent(message: str):
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
    
    if best_agent == "concordia":
        return concordia_agent.concordia_agent
    elif best_agent == "ai":
        return ai_agent.ai_agent
    else:
        return general_agent.general_agent

@app.post("/chat")
def chat(request: ChatRequest):
    # Retrieve relevant context from ChromaDB for the current session using the user's message as query.
    previous_contexts = get_context(request.session_id, request.message, n_results=3)
    # print("Retrieved Context:", previous_contexts)
    context_text = "\n".join(previous_contexts) + "\n" if previous_contexts else ""
    
    # Build the full prompt by pre-pending the retrieved context.
    full_prompt = f"{context_text}User: {request.message}\nAssistant:"
    
    # Choose the agent based on the message content.
    agent_func = choose_agent(request.message)
    response = agent_func(full_prompt)
    
    # Save the current user message and assistant response in ChromaDB.
    add_context(request.session_id, request.message, role="user")
    add_context(request.session_id, response, role="assistant")
    
    return {"message": response}
