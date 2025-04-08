import requests
import json

"""
This script demonstrates a simple chatbot using FastAPI and LangChain.
The chatbot maintains a conversation session with the server.
"""

API_URL = "http://localhost:8000/chat"

print("Welcome to the Chatbot! Type 'exit' to quit.")

# Store the session ID between requests
session_id = None

while True:
    # Accept user input and send it to the API
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Include session_id if we have one from a previous response
    payload = {"message": user_input}
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        bot_response = data["message"]
        # Update session_id for next request
        session_id = data["session_id"]
        print("Chatbot:", bot_response)
    else:
        print("Error:", response.status_code)
        print(response.text)