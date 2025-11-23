# models/prompts.py 


from typing import List

# models/prompts.py

# models/prompts.py
ROUTER_SYSTEM_PROMPT = """You are a small intent router AND replier.

Return ONLY one JSON object with these keys:
- "intent": one of ["greeting","acknowledgement","out_of_scope","other"]
- "action": one of ["answer_here","continue"]
- "assistant_reply": string (required when intent is "greeting", "acknowledgement", or "out_of_scope")

Rules:
- Treat small-talk as greeting: "hi", "hello", "good morning", "how are you", etc.
- Acknowledgement = thanks/ok/got it/that helps.
- If the message is general/non-DNV (e.g., weather/temperature/forecast, stock market/bitcoin/crypto, time now, sports score, capitals, population, age/how old), set intent="out_of_scope" and write a brief reply (≤ {max_words} words) stating you assist only with DNV healthcare standards.
- If the message mentions DNV standards/programs (e.g., NIAHO, HOSP, CAH, BEHAV, LAB, QM.x, IC.x, CC.x), set intent="other".
- When intent is greeting/acknowledgement/out_of_scope, write a friendly reply ≤ {max_words} words.
- Output must be valid JSON with double quotes only. No markdown, no code fences, no commentary.
- Output exactly ONE JSON object."""



"""
DNV Healthcare Standards Assistant Prompt Templates
ENHANCED with clarification prompts and program-aware messaging
"""

# System prompt for DNV healthcare standards
SYSTEM_PROMPT = """You are an expert assistant specializing in DNV healthcare standards, 
NIAHO accreditation requirements, and healthcare quality compliance. Your role is to provide 
accurate, actionable guidance based on official DNV standards and best practices.

When answering questions:
1. Prioritize accuracy and cite specific standards when applicable
2. Provide practical, implementable advice
3. Be concise but comprehensive
4. If the context doesn't contain sufficient information, acknowledge the limitation
5. Focus on compliance and quality improvement

Remember: You are a trusted advisor for healthcare facilities seeking DNV accreditation and compliance."""


# Main QA prompt template
QA_PROMPT_TEMPLATE = """Based on the following context from DNV standards and documentation, 
please answer the user's question.

Context:
{context}

User Question: {question}

Please provide a comprehensive answer based on the DNV standards above. If the context doesn't 
provide enough information to fully answer the question, indicate what additional information 
would be needed and suggest contacting DNV directly at healthcare@dnv.com for detailed guidance.

Answer:"""


"""
ENHANCED: Program-aware prompt template
Used when we know which program the user is asking about
Provides more focused responses by emphasizing the specific program context
"""
QA_PROMPT_WITH_PROGRAM_TEMPLATE = """You are assisting with {program} standards and requirements.

Context from {program} Documentation:
{context}

User Question: {question}

Please provide a comprehensive answer specific to {program} based on the documentation above. Include:
1. Direct answer to the question with specific requirements
2. References to relevant standard sections
3. Practical implementation guidance for {program} compliance
4. Any critical compliance considerations

If the context doesn't provide sufficient information, suggest contacting DNV directly at healthcare@dnv.com.

Answer:"""


"""
ENHANCED: Low confidence clarification prompt
Used when search results have low relevance scores
Helps narrow down the query to get better results
"""
LOW_CONFIDENCE_CLARIFICATION_TEMPLATE = """
Your question: "{query}"
{clarification_questions}

This will help me provide more specific and accurate guidance from the DNV standards."""


"""
ENHANCED: Program clarification prompt
Used when program cannot be determined from query or context
"""
PROGRAM_CLARIFICATION_TEMPLATE = """I'd be happy to help with your DNV standards question!

Your question: "{query}"

To provide the most accurate information, could you please specify the program?
"""


"""
Helper function to build clarification questions based on query analysis
"""
def build_low_confidence_questions(query: str, chapter_hints: List[str]) -> str:
    """
    Generate relevant clarification questions based on query content
    
    Args:
        query: Original user query
        chapter_hints: Detected chapter/domain hints
        
    Returns:
        Formatted clarification questions string
    """
    questions = []
    
    if not chapter_hints:
        questions.append("• Which specific standard or chapter are you asking about? (e.g., QM.1, CC.1, IC.5)")
    
    query_lower = query.lower()
    
    if "requirement" in query_lower and not any(word in query_lower for word in ["implement", "comply", "meet"]):
        questions.append("• Are you asking about the base requirements, interpretive guidelines, or implementation strategies?")
    
    if any(word in query_lower for word in ["policy", "policies", "procedure"]):
        questions.append("• Are you asking about creating a new policy, updating an existing policy, or reviewing policy compliance?")
    
    #if not questions:
    #    questions.append("• Could you provide more details about what specific aspect you'd like to know?")
    #    questions.append("• Are you looking for requirements, implementation guidance, or compliance verification?")
    
    if not questions:
        questions.append("• Sorry, I couldn’t find the information you’re looking for. Can you be more specific?")
        
    
    return "\n".join(questions)


"""
Helper function to format program suggestions
"""
def format_program_suggestions(programs: List[str]) -> str:
    """
    Format program list for clarification prompt
    
    Args:
        programs: List of valid program codes
        
    Returns:
        Formatted program list with descriptions
    """
    program_descriptions = {
        "NIAHO-HOSP": "NIAHO Hospital Accreditation",
        "NIAHO-CAH": "NIAHO Critical Access Hospital",
        "NIAHO-PSY": "NIAHO Psychiatric Hospital",
        "NIAHO-PROC": "NIAHO Procedural Sedation",
        "CARDIAC_VASCULAR": "Cardiac and Vascular Certification",
        "STROKE": "Stroke Center Certification",
        "INFECTION": "Infection Prevention and Control",
        "CYBERSECURITY": "Cybersecurity Certification",
        "GLYCEMIC": "Glycemic Control Certification",
        "ORTHO_SPINE": "Orthopedic and Spine Certification",
        "PALLIATIVE": "Palliative Care Certification",
        "EXTRACORPOREAL": "Extracorporeal Life Support",
        "VENTRICULAR": "Ventricular Assist Device",
        "SURVEY": "Survey Process"
    }
    
    suggestions = []
    for program in programs:
        description = program_descriptions.get(program, program)
        suggestions.append(f"• {description} ({program})")
    
    return "\n".join(suggestions)


def get_chat_prompt():
    """Create the chat prompt template for RAG."""
    return {
        "system": SYSTEM_PROMPT,
        "template": QA_PROMPT_TEMPLATE
    }


def get_simple_prompt():
    """Create a simple prompt without system message for nova-micro."""
    return QA_PROMPT_TEMPLATE


def get_program_aware_prompt():
    """Get program-aware prompt template"""
    return QA_PROMPT_WITH_PROGRAM_TEMPLATE


def get_low_confidence_prompt():
    """Get low confidence clarification template"""
    return LOW_CONFIDENCE_CLARIFICATION_TEMPLATE


def get_program_clarification_prompt():
    """Get program clarification template"""
    return PROGRAM_CLARIFICATION_TEMPLATE