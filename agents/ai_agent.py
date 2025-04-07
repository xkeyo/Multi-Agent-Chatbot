from ollama_service import ask_ollama

def ai_agent(message: str) -> str:
    # Comprehensive additional context for AI-related questions.
    additional_context = """
Additional Context for AI-Related Questions:
- Artificial Intelligence (AI) is a broad field of computer science dedicated to creating systems that can perform tasks that typically require human intelligence. AI encompasses the simulation of human cognitive functions such as learning, reasoning, problem-solving, perception, and language understanding.
- Historically, AI research began with symbolic systems and rule-based approaches in the 1950s. Over time, the field evolved to include statistical methods and, more recently, machine learning techniques, particularly deep learning, which uses multi-layered neural networks to analyze large amounts of data.
- Key subfields of AI include:
    • Machine Learning (ML): Focuses on developing algorithms that allow computers to learn patterns from data. ML techniques are divided into supervised, unsupervised, and reinforcement learning, with deep learning being a prominent subset.
    • Natural Language Processing (NLP): Enables machines to understand, interpret, and generate human language. This area underpins technologies such as chatbots, translation systems, and sentiment analysis.
    • Computer Vision: Deals with enabling machines to interpret and understand visual information from the world, such as image and video recognition.
    • Robotics: Involves designing and building robots that can perform tasks autonomously or semi-autonomously, often incorporating AI to enhance decision-making and adaptability.
    • Expert Systems: AI systems that emulate the decision-making abilities of human experts by using knowledge bases and inference rules.
- Modern breakthroughs in AI include the advent of transformer models (such as GPT series), which have revolutionized NLP, and notable successes in reinforcement learning (e.g., AlphaGo and similar systems). These advances have significantly impacted both research and practical applications.
- AI applications are widespread across industries, including healthcare (for diagnostics and personalized treatment), finance (for fraud detection and algorithmic trading), autonomous vehicles, customer service, and more.
- Alongside technical progress, ethical considerations are paramount. Topics such as fairness, bias, transparency, privacy, and the societal impact of AI are critically important as these technologies become more integrated into daily life.
- Researchers continue to explore areas like explainable AI (XAI) to make AI decision processes more transparent, and AI safety to ensure that advanced systems are secure and aligned with human values.
"""
    prompt = f"""You are an expert in Artificial Intelligence with deep knowledge across theory, research, and practical applications. Use the following background information for internal context to provide a clear, detailed, and accurate answer to any AI-related questions, but do not include the background text verbatim in your final response.

Background Information (for internal use):
{additional_context}

User: {message}
Assistant:"""
    return ask_ollama(prompt)
