from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent
from context_store import add_context
from sentence_transformers import SentenceTransformer, util
from external.wiki_search import search_wikipedia
import langchain_memory as lch


app = FastAPI()

class ChatRequest(BaseModel):
    message: str

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

def choose_agent(message: str):
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
    
    return best_agent

def extract_query(message: str) -> str:
    """Extract a concise topic from the user's message."""
    # If message starts with "User:", remove it.
    if message.lower().startswith("user:"):
        message = message[5:].strip()
    
    # Remove any trailing "Assistant:" parts if present.
    if "assistant:" in message.lower():
        message = message.lower().split("assistant:")[0].strip()
    
    # Define common introductory phrases to remove
    prefixes = [
        "tell me about",
        "what is",
        "explain",
        "give me information on",
        "info on",
        "information on"
    ]
    query = message.lower().strip()
    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()
            break
    # Remove trailing punctuation
    query = query.rstrip("?.!")
    return query

@app.post("/chat")
def chat(request: ChatRequest):
    # Determine the appropriate agent type
    agent_type = choose_agent(request.message)
    
    # Get wiki info for general agent
    wiki_info = ""
    if agent_type == "general":
        wiki_query = extract_query(request.message)
        wiki_info = search_wikipedia(wiki_query)
    
    # Generate an enhanced prompt using LangChain
    enhanced_prompt = lch.generate_prompt(
        agent_type=agent_type,
        message=request.message,
        wiki_info=wiki_info
    )
    
    # Call the appropriate agent function
    if agent_type == "concordia":
        response = concordia_agent.concordia_agent(enhanced_prompt)
    elif agent_type == "ai":
        response = ai_agent.ai_agent(enhanced_prompt)
    else:
        response = general_agent.general_agent(enhanced_prompt)
    
    # Save the conversation in LangChain memory
    lch.add_to_memory(request.message, response)
    
    # Also save in ChromaDB for possible vector search later
    add_context("default", request.message, role="user")
    add_context("default", response, role="assistant")
    
    return {"message": response}