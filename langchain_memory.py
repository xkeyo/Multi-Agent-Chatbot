from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from context_store import add_context as db_add_context, get_context as db_get_context

# Session memory dictionary
session_memories = {}

def get_session_memory(session_id: str):
    """Get or create a LangChain memory object for the session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=True)
    return session_memories[session_id]

def add_message_to_memory(session_id: str, message: str, role: str):
    """Add a message to both ChromaDB and LangChain memory"""
    # Add to ChromaDB for vector retrieval
    db_add_context(session_id, message, role)
    
    # Add to LangChain memory
    memory = get_session_memory(session_id)
    if role == "user":
        memory.chat_memory.add_user_message(message)
    else:
        memory.chat_memory.add_ai_message(message)

def get_prompt_template(template_type="general"):
    """Get a LangChain prompt template based on the type"""
    templates = {
        "general": PromptTemplate(
            input_variables=["context", "history", "message"],
            template="""
You are a helpful AI assistant. Please provide a thoughtful response based on the following:

Previous context:
{context}

Conversation history:
{history}

User: {message}
Assistant:"""
        ),
        "ai": PromptTemplate(
            input_variables=["context", "history", "message", "ai_context"],
            template="""
You are an expert in Artificial Intelligence. Please provide a detailed and informative response based on the following:

Previous context:
{context}

Additional AI context:
{ai_context}

Conversation history:
{history}

User: {message}
Assistant:"""
        ),
        "concordia": PromptTemplate(
            input_variables=["context", "history", "message", "concordia_context"],
            template="""
You are a Concordia University admissions specialist for the Computer Science program. Please provide helpful information based on the following:

Previous context:
{context}

Concordia CS program information:
{concordia_context}

Conversation history:
{history}

User: {message}
Assistant:"""
        )
    }
    
    return templates.get(template_type, templates["general"])

def get_formatted_history(session_id: str):
    """Get formatted conversation history from LangChain memory"""
    memory = get_session_memory(session_id)
    messages = memory.chat_memory.messages
    
    if not messages:
        return ""
    
    formatted_history = ""
    for msg in messages[-6:]:  # Limit to last 6 messages to keep context manageable
        if msg.type == "human":
            formatted_history += f"User: {msg.content}\n"
        else:
            formatted_history += f"Assistant: {msg.content}\n"
    
    return formatted_history