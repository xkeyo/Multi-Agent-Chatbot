from fastapi import FastAPI
import ollama

# Create a FastAPI instance
app = FastAPI()

@app.post("/chat")
def chat(message: str):
    response = ollama.chat(model = "llama2", prompt=message)
    return {"message": response}