from ollama_service import ask_ollama


def concordia_agent(message : str) -> str:

    prompt = f""" You're an expert in Concordia University Computer Science Admissions. 
    You can answer questions, provide information, and assist with various tasks related to 
    Concordia University Computer Science Admissions.

    User: {message}
    Assistant: """

    # Call the ollama_chat function with the prompt
    return ask_ollama(prompt)