# generator.py
"""Website Intelligence AI - Answer Generation Module

Generates answers to user queries based on retrieved context from the knowledge base.
"""

from llm_model import llm
from utils.logger import get_logger

logger = get_logger("GENERATOR")

# ============================================================
# CONTEXT VALIDATION
# ============================================================

def has_sufficient_context(context: str) -> bool:
    """Checks if the retrieved context is sufficient to attempt an answer.
    
    This is a simple heuristic based on length, but could be enhanced with 
    more complex logic like relevance scoring or keyword matching.
    
    Args:
        context: The retrieved context string
        
    Returns:
        True if context is sufficient, False otherwise
    """
    if not context or len(context.strip()) == 0:
        return False
    return len(context) > 50

# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(query: str, context: str) -> str:
    """Builds the LLM prompt for answer generation.
    
    Combines the user query with retrieved context and safety instructions
    to guide the LLM to generate accurate, sourced answers.
    
    Args:
        query: User's original question
        context: Retrieved context from the knowledge base
        
    Returns:
        Complete prompt string ready for LLM invocation
    """
    prompt = f"""
        You are an assistant that answers questions based on the following retrieved context from a company's website. 
        Use ONLY the information in the context to answer the question. If you don't know the answer, say you don't know. 
        Do NOT make up answers.

        Context:
        {context}

        OriginalQuestion: {query}

        Using ONLY the above context, provide a concise and accurate answer to the question. If the context does not contain the answer, respond with "Sorry, I don't have enough information to answer that question."

    """
    return prompt

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate(query: str, retrieval_result: dict) -> dict:
    """Generates an answer to the user query based on retrieved context.
    
    Steps: 
    1. Check if retrieved context is sufficient
    2. If sufficient, build a prompt with the context and query
    3. Call LLM to generate an answer
    4. Return the answer along with sources and original query for reference
    
    Args:
        query: User's original question
        retrieval_result: Dictionary containing retrieved context and sources
        
    Returns:
        Dictionary with has_answer, answer, sources, and metadata
    """
    context = retrieval_result.get("context", "")
    
    if not has_sufficient_context(context):
        logger.warning("Insufficient context for query")
        return {
            "has_answer": False,
            "answer": "Sorry, I don't have enough information to answer that question.",
            "sources": retrieval_result.get("sources", [])
        }
    
    logger.debug("Sufficient context available, building prompt")
    prompt = build_prompt(query, context)
    
    try:
        logger.debug("Invoking LLM for answer generation")
        answer = llm.invoke(prompt).content
        logger.info("Answer generated successfully")
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return {
            "has_answer": False,
            "answer": f"Generation failed: {e}",
            "sources": retrieval_result.get("sources", []),
            "original_query": query,
        }
    
    return {
        "has_answer": True,
        "answer": answer,
        "sources": retrieval_result.get("sources", []),
        "original_query": query,
    }