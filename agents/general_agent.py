from ollama_service import ask_ollama
from external.wiki_search import search_wikipedia
import logging

def extract_query(message: str) -> str:
    """
    Extract a concise topic from the user's message.
    This function removes known prefixes and any extraneous parts.
    """
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

def general_agent(message: str) -> str:
    # Attempt to extract a concise query from the original user input.
    # Assume the original user message is in the first line, after "User:" if present.
    lines = message.splitlines()
    if lines and lines[0].lower().startswith("user:"):
        original_query = lines[0][len("user:"):].strip()
    else:
        original_query = message
    wiki_query = extract_query(original_query)
    # logging.debug(f"Extracted query for Wikipedia: '{wiki_query}'")
    
    wiki_info = search_wikipedia(wiki_query)
    
    prompt = f"""You're a general AI assistant. You can answer questions, provide information, and assist with various tasks.

Here is some background information from Wikipedia on "{wiki_query}":
{wiki_info}

User: {message}
Assistant:"""
    return ask_ollama(prompt)
