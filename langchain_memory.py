# langchain_integration.py
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from typing import Dict, List, Optional

class LangChainMemoryManager:
    def __init__(self, session_id: str = "default", memory_type: str = "buffer", k: int = 5):
        """
        Initialize the LangChain memory manager.
        
        Args:
            session_id: Unique identifier for the conversation session
            memory_type: Type of memory to use ("buffer" or "window")
            k: Number of previous exchanges to keep in window memory
        """
        self.session_id = session_id
        self.memory_type = memory_type
        
        # Initialize Ollama model
        self.llm = Ollama(model="llama3.2")
        
        # Choose memory type
        if memory_type == "window":
            self.memory = ConversationBufferWindowMemory(k=k)
        else:  # Default to buffer memory
            self.memory = ConversationBufferMemory()
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=False
        )
    
    def get_conversation_history(self) -> str:
        """Get the current conversation history from memory"""
        return self.memory.buffer
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory"""
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI response to memory"""
        self.memory.chat_memory.add_ai_message(message)
    
    def generate_response(self, message: str, custom_prompt: Optional[str] = None) -> str:
        """Generate a response using the conversation chain"""
        if custom_prompt:
            # Use custom prompt template if provided
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=custom_prompt
            )
            self.conversation.prompt = prompt
        
        return self.conversation.predict(input=message)
    
    def clear_memory(self) -> None:
        """Clear the conversation memory"""
        self.memory.clear()


class PromptEngineering:
    """Helper class for advanced prompt engineering techniques"""
    
    @staticmethod
    def create_domain_specific_prompt(domain: str, user_input: str, context: str = "") -> str:
        """Create a domain-specific prompt template"""
        
        templates = {
            "ai": """
You are an expert in Artificial Intelligence with deep knowledge across theory, research, and practical applications.
Use your expertise to provide a clear, detailed, and accurate answer to the question.

Context information:
{context}

Conversation history:
{history}

User question: {input}
AI Assistant:""",
            
            "concordia": """
You are an expert in Concordia University Computer Science Admissions with a deep understanding of the program structure, 
admission criteria, co-op opportunities, and career outcomes.

Context information:
{context}

Conversation history:
{history}

User question: {input}
AI Assistant:""",
            
            "general": """
You're a helpful AI assistant that provides accurate, concise, and relevant information on a wide range of topics.

Context information:
{context}

Conversation history:
{history}

User question: {input}
AI Assistant:"""
        }
        
        # Default to general if domain not found
        template = templates.get(domain, templates["general"])
        
        # Create the prompt template
        return template.format(context=context, history="", input=user_input)
    
    @staticmethod
    def create_pipeline_prompt(components: List[Dict]) -> PipelinePromptTemplate:
        """Create a pipeline prompt from multiple components"""
        prompt_templates = []
        final_variables = set()
        
        for component in components:
            template = PromptTemplate(
                template=component["template"],
                input_variables=component["variables"]
            )
            prompt_templates.append({"name": component["name"], "prompt": template})
            final_variables.update(component["variables"])
        
        # The final template combines all components
        final_template = "\n\n".join([f"{{{component['name']}}}" for component in components])
        
        return PipelinePromptTemplate(
            final_prompt=PromptTemplate(template=final_template, input_variables=list(final_variables)),
            pipeline_prompts=prompt_templates
        )