# generator.py
from llm_model import llm


def has_sufficient_context(context:str) -> bool:

    """Checks if the retrieved context is sufficient to attempt an answer.
    This is a simple heuristic based on length, but could be enhanced with more complex logic.
    """
    if not context or len(context.strip()) == 0:
        return False
    return len(context) > 50




def build_prompt(query:str, context:str) -> str:
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



def generate(query:str, retrieval_result:dict) -> dict:

    """Generates an answer to the user query based on retrieved context.
    Steps: 
    1. Check if retrieved context is sufficient
    2. If sufficient, build a prompt with the context and query
    3. Call LLM to generate an answer
    4. Return the answer along with sources and original query for reference"""

    context = retrieval_result.get("context", "")
    
    if not has_sufficient_context(context):
        return {
            "has_answer": False,
            "answer": "Sorry, I don't have enough information to answer that question.",
            "sources": retrieval_result.get("sources", [])
        }
    
    prompt = build_prompt(query, context)
    
    try:
        answer = llm.invoke(prompt).content
    except Exception as e:
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