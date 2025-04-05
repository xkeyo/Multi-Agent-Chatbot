from fastapi import FastAPI
from pydantic import BaseModel
from agents import ai_agent, concordia_agent, general_agent
from context_store import add_context, get_context

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # default session if none provided

def choose_agent(message: str):
    lower_message = message.lower()
    # Only choose Concordia agent if the message explicitly mentions "concordia", "admission", and "computer science"
    if "concordia" in lower_message and "admission" in lower_message and "computer science" in lower_message:
        return concordia_agent.concordia_agent
    elif "ai" in lower_message or "machine learning" in lower_message:
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
