# Adaptive Multi-Agent Chatbot System Using Ollama

This project aims to develop an adaptive multi-agent chatbot system that leverages Ollama to provide context-aware assistance. The chatbot system is built on a multi-agent architecture where each agent specializes in a particular area.  The system is made to answer a variety of queries, such as general inquiries, questions concerning artificial intelligence, and inquiries about admissions to Concordia University's computer science program. The system combines context awareness via a vector-based memory store using ChromaDB and also integrates an external knowledge integration using Wikipedia in addition to basic answer production.

- **General Agent:** Agent that handles general questions with a context help from Wikipedia (See implementation in general_agent.py).
- **Concordia Agent:** Agent that is dedicated to inquiries related to Concordia University Computer Science admission (See implementation in concordia_agent.py).
- **AI Agent:** Agent that specializes in artificial intelligence and machine learning topics (See implementation in ai_agent.py).


---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)

---

## Features

- **Multi-Agent Architecture:**  
  Three specialized agents handle different domains, ensuring targeted and relevant responses.

- **Context Management:**  
  Utilizes ChromaDB to store conversation history and supports multi-turn interactions with vector embeddings from SentenceTransformer.

- **External API Integration:**  
  - **Ollama Service API:** Generates natural language responses based on constructed prompts.
  - **Wikipedia API:** Fetches concise summaries to provide additional background context for improved accuracy.

- **FastAPI Server:**  
  Acts as the backend interface to handle user requests, context retrieval, and agent dispatching.

---

## Project Structure

```plaintext
.
├── chatbot.py                # Terminal interface to interact with the chatbot.
├── context_store.py          # Manages conversation context storage using ChromaDB.
├── main.py                   # FastAPI server handling API endpoints and agent selection.
├── ollama_service.py         # Handles communication with the Ollama.
├── wiki_search.py            # Module for querying Wikipedia.
├── ai_agent.py               # Agent for handling AI-related questions.
├── concordia_agent.py        # Agent for handling Concordia admissions inquiries.
├── general_agent.py          # Agent for handling general inquiries with Wikipedia integration.
└── requirements.txt          # List of Python dependencies.

```

---

## Instalation
- Python 3.8+
- pip (Python package installer)
- Access to the following external services:
- Ollama Service: Running on http://localhost:11434/api/generate
- ChromaDB: Configured and accessible locally or remotely as needed

### Clone Reposittory
```plaintext

git clone https://github.com/xkeyo/COMP-474-Project2
cd COMP-474-Project2

```

### Create and Activate a Virtual Environment:
```plaintext

python -m venv venv

```

#### On macOS/Linux:
```plaintext

source venv/bin/activate

```

#### On Windows:
```plaintext

venv\Scripts\activate

```

### Install Dependencies:
```plaintext

pip install -r requirements.txt

```

### Configure External Services:
- Ensure the Ollama service is running at http://localhost:11434/api/generate.
- Set up and run ChromaDB.

--- 

## Usage
### Running the FastAPI Server
Start the FastAPI server using Uvicorn:

```plaintext

uvicorn main:app --reload

```
The server will be accessible at http://localhost:8000.

### Command-Line Chat Interface
Run the terminal interface to interact with the chatbot:
```plaintext

python chatbot.py

```

Type your message and press Enter. Type "exit" to quit the session.

---
## How it Works
#### 1. UseInput & API Request:
The terminal interface sends user input to the FastAPI server.

#### 2. Context Retrieval:
The server queries ChromaDB to retrieve relevant past conversation context for the current session.

#### 3. Prompt Construction & Agent Selection:
A prompt is built from the retrieved context and the user’s current input. The system selects the appropriate agent based on vector similarity comparisons and keywords.

#### 4. Response Generation:
The chosen agent constructs a prompt for the Ollama API, which generates a response.

#### 5. Context Update:
Both the user’s query and the assistant’s response are stored in ChromaDB to maintain conversation context for future interactions.


