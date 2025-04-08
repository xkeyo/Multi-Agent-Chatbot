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
    
    # Return agent type and function
    if best_agent == "concordia":
        return "concordia", concordia_agent.concordia_agent
    elif best_agent == "ai":
        return "ai", ai_agent.ai_agent
    else:
        return "general", general_agent.general_agent

@app.post("/chat")
def chat(request: ChatRequest):
    # Retrieve relevant context from ChromaDB for the current session using the user's message as query
    previous_contexts = get_context(request.session_id, request.message, n_results=3)
    context_text = "\n".join(previous_contexts) if previous_contexts else "No relevant context found."
    
    # Get conversation history from LangChain memory
    conversation_history = get_formatted_history(request.session_id)
    
    # Choose the agent based on the message content
    agent_type, agent_func = choose_agent(request.message)
    
    # Get the appropriate prompt template based on agent type
    prompt_template = get_prompt_template(agent_type)
    
    # Prepare prompt variables
    prompt_vars = {
        "context": context_text,
        "history": conversation_history,
        "message": request.message
    }
    
    # Add agent-specific context if needed
    if agent_type == "ai":
        ai_additional_context = """
        AI encompasses machine learning, neural networks, computer vision, natural language processing,
        robotics, and many other subfields. Recent breakthroughs include large language models, diffusion models
        for image generation, and reinforcement learning for complex decision-making.
        """
        prompt_vars["ai_context"] = ai_additional_context
    
    elif agent_type == "concordia":
        concordia_additional_context = """
        Concordia University's Computer Science program offers Bachelor's, Master's, and PhD degrees.
        Admission requirements include strong math skills, with CEGEP students needing a 27+ overall average
        and 26+ in math courses. The program covers programming, algorithms, data structures, AI, and software engineering.
        """
        prompt_vars["concordia_context"] = concordia_additional_context
    
    # Format the prompt using LangChain's template
    formatted_prompt = prompt_template.format(**prompt_vars)
    
    # Get response from the appropriate agent
    response = agent_func(formatted_prompt)
    
    # Save the user message and bot response using LangChain memory and ChromaDB
    add_message_to_memory(request.session_id, request.message, role="user")
    add_message_to_memory(request.session_id, response, role="assistant")
    
    return {"message": response}