from ollama_service import ask_ollama

def general_agent(message : str) -> str:

    prompt = f""" You're a general AI assistant. You can answer questions, provide information, and assist with various tasks.

    User: {message}
    Assistant: """

    # Call the ollama_chat function with the prompt
    return ask_ollama(prompt)