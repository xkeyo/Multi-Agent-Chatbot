from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent

app = FastAPI()

# In-memory conversation context (session_id -> list of previous interactions)
session_context = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # default session if none provided

def choose_agent(message: str):
    lower_message = message.lower()
    if "concordia" in lower_message and "admission" in lower_message:
        return concordia_agent.concordia_agent
    elif "ai" in lower_message or "machine learning" in lower_message:
        return ai_agent.ai_agent
    else:
        return general_agent.general_agent

@app.post("/chat")
def chat(request: ChatRequest):
    # Retrieve the last 3 context entries if available
    context = ""
    if request.session_id in session_context:
        history = session_context[request.session_id]
        context = "\n".join(history[-3:]) + "\n" if history else ""
    
    # Build the full prompt including any context
    full_prompt = f"{context}User: {request.message}\nAssistant:"
    
    # Choose the appropriate agent based on message content
    agent_func = choose_agent(request.message)
    response = agent_func(full_prompt)
    
    # Update conversation context for multi-turn conversation
    session_context.setdefault(request.session_id, []).append(f"User: {request.message}\nAssistant: {response}")
    
    return {"message": response}
