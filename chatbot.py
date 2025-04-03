import requests

"""
This script demonstrates a simple chatbot using FastAPI and Ollama.
The chatbot listens for user input, sends it to the FastAPI server,
and prints the response from the server.
"""

API_URL = "http://localhost:8000/chat"

print("Welcome to the Chatbot! Type 'exit' to quit.")

while True:
    # Accept user input and send it to the API
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = requests.post(API_URL, json={"message": user_input})

    if response.status_code == 200:
        bot_response = response.json()["message"]
        print("Chatbot:", bot_response)
    else:
        print("Error:", response.status_code)
