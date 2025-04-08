from langchain.memory import ConversationBufferMemory
import logging

memory = ConversationBufferMemory(return_messages=True)

def add_to_memory(user_message: str, ai_response: str):
    """Add a conversation turn to memory"""
    memory.save_context({"input": user_message}, {"output": ai_response})

def get_conversation_history() -> str:
    """Get formatted conversation history"""
    try:
        memory_vars = memory.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        # Format the messages into a string
        history = ""
        for i in range(0, len(messages), 2):
            if i < len(messages):
                user_msg = messages[i].content
                history += f"User: {user_msg}\n"
            
            if i+1 < len(messages):
                ai_msg = messages[i+1].content
                history += f"Assistant: {ai_msg}\n"
        
        return history
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {e}")
        return ""

def clear_memory():
    """Clear the conversation memory"""
    memory.clear()

# Define prompt templates for each agent type
PROMPT_TEMPLATES = {
    "general": """You're a general AI assistant. You can answer questions, provide information, and assist with various tasks.

Previous conversation:
{conversation_history}

Here is some background information from Wikipedia that might be relevant:
{wiki_info}

User: {message}
Assistant:""",

    "ai": """You are an expert in Artificial Intelligence with deep knowledge across theory, research, and practical applications.

Background Information (for internal context):
- Artificial Intelligence (AI) is a broad field of computer science dedicated to creating systems that can perform tasks that typically require human intelligence.
- Key subfields include Machine Learning, Natural Language Processing, Computer Vision, Robotics, and Expert Systems.
- Modern breakthroughs in AI include transformer models and reinforcement learning advances.
- AI applications span healthcare, finance, autonomous vehicles, customer service, and more.
- Ethical considerations include fairness, bias, transparency, privacy, and societal impact.

Previous conversation:
{conversation_history}

User: {message}
Assistant:""",

    "concordia": """You are an expert in Concordia University Computer Science Admissions with a deep understanding of the program structure, admission criteria, co-op opportunities, and career outcomes.

Background Information (for internal context):
- The Bachelor of Computer Science (BCompSc) is offered by Concordia University through the Gina Cody School of Engineering and Computer Science.
- For Quebec CEGEP applicants, admission requires an overall average of 27 and a minimum math average of 26.
- High school applicants need to achieve an A- overall with strong math performance.
- The program offers co-op opportunities for practical industry experience.
- Graduates pursue careers in healthcare, communications, finance, manufacturing, and technology sectors.

Previous conversation:
{conversation_history}

User: {message}
Assistant:"""
}

def generate_prompt(agent_type: str, message: str, wiki_info: str = "") -> str:
    """Generate an enhanced prompt using LangChain"""
    # Get conversation history
    conversation_history = get_conversation_history()
    
    # Select the appropriate template
    template = PROMPT_TEMPLATES.get(agent_type, PROMPT_TEMPLATES["general"])
    
    # Create the prompt template
    if agent_type == "general":
        prompt = template.format(
            conversation_history=conversation_history,
            wiki_info=wiki_info,
            message=message
        )
    else:
        prompt = template.format(
            conversation_history=conversation_history,
            message=message
        )
    
    return prompt