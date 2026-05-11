# generator.py
"""Website Intelligence AI - Answer Generation Module

Generates answers to user queries based on retrieved context from the knowledge base.
"""

from llm_model import llm
from utils.logger import get_logger
from langsmith import traceable

logger = get_logger("GENERATOR")

# ============================================================
# SUGGESTED QUESTIONS PER COMPANY TYPE
# ============================================================

SUGGESTED_QUESTIONS = {
    "docs":      ["How do I install it?", "What are the main features?", "Show me a quick start example"],
    "tech_docs": ["How do I get started?", "What is the API reference?", "Show me a configuration example"],
    "ecommerce": ["What products do you offer?", "What are your shipping options?", "How do I return an item?"],
    "blog":      ["What topics do you cover?", "Who writes for this blog?", "What are your most popular posts?"],
    "support":   ["How do I contact support?", "What are the most common issues?", "How do I reset my password?"],
    "default":   ["What does this website offer?", "How can I get started?", "Who is this for?"],
}

def get_suggestions(company_type: str) -> list[str]:
    return SUGGESTED_QUESTIONS.get(company_type, SUGGESTED_QUESTIONS["default"])


# ============================================================
# QUERY CLASSIFICATION
# ============================================================

GREETING_WORDS = {"hi", "hello", "hey", "howdy", "hiya", "greetings", "sup", "yo"}

def is_greeting(query: str) -> bool:
    q = query.lower().strip().rstrip("!?.,").strip()
    words = set(q.split())
    return bool(words & GREETING_WORDS) and len(q.split()) <= 4

def is_capability_query(query: str) -> bool:
    q = query.lower().strip()
    triggers = [
        "what can you do", "what can you help", "what do you know",
        "what can i ask", "what are you", "what do you do",
        "capabilities", "what can you tell", "how can you help",
        "what can you help me with",
    ]
    return any(t in q for t in triggers)


# ============================================================
# SPECIAL RESPONSES
# ============================================================

def greeting_response(company_id: str, company_type: str) -> dict:
    suggestions = get_suggestions(company_type)
    answer = f"Hi! I'm ready to answer questions about **{company_id}**. I've indexed its website content so you can ask me anything about it."
    return {
        "has_answer": True,
        "answer": answer,
        "sources": [],
        "follow_ups": suggestions,
        "original_query": "",
    }

def capability_response(company_id: str, company_type: str) -> dict:
    suggestions = get_suggestions(company_type)
    answer = f"I can answer questions about **{company_id}** based on its website content. I've crawled and indexed the site, so I know about its features, documentation, and more. Here are some things you can ask:"
    return {
        "has_answer": True,
        "answer": answer,
        "sources": [],
        "follow_ups": suggestions,
        "original_query": "",
    }

def out_of_scope_response(company_id: str, company_type: str, sources: list) -> dict:
    suggestions = get_suggestions(company_type)
    answer = f"I couldn't find relevant information about that in the **{company_id}** content I've indexed. Try asking something more specific to the site."
    return {
        "has_answer": False,
        "answer": answer,
        "sources": sources,
        "follow_ups": suggestions,
        "original_query": "",
    }


# ============================================================
# CONTEXT VALIDATION
# ============================================================

def has_sufficient_context(context: str) -> bool:
    if not context or len(context.strip()) == 0:
        return False
    return len(context) > 50


# ============================================================
# PROMPT BUILDING
# ============================================================

SYSTEM_PROMPT = """You are an assistant that answers questions based on retrieved context from a company's website.

Rules:
- Use ONLY the information in the context to answer
- If you don't know the answer, say you don't know
- Do NOT make up answers
- Be concise and helpful

After your answer, on a new line, always add exactly this line:
SUGGESTIONS: [follow-up question 1] | [follow-up question 2]

The follow-up questions should be natural next questions a user might ask based on your answer."""


def build_prompt(query: str, messages: list[dict], context: str) -> list:
    messages = messages or []
    prompt = [("system", SYSTEM_PROMPT)]
    for message in messages:
        prompt.append((message["role"], message["content"]))
    prompt.append(("user", query + "\n\nContext:\n" + context))
    return prompt


def parse_follow_ups(answer: str) -> tuple[str, list[str]]:
    """Parse follow-up suggestions from LLM response."""
    if "SUGGESTIONS:" in answer:
        parts = answer.split("SUGGESTIONS:")
        clean_answer = parts[0].strip()
        suggestions_text = parts[1].strip()
        follow_ups = [q.strip() for q in suggestions_text.split("|") if q.strip()][:3]
        return clean_answer, follow_ups
    return answer, []


# ============================================================
# ANSWER GENERATION
# ============================================================

@traceable(name="generate")
def generate(
    query: str,
    messages: list[dict] | None,
    retrieval_result: dict,
    company_id: str = "",
    company_type: str = "default"
) -> dict:
    """Generates an answer to the user query based on retrieved context.

    Args:
        query: User's original question
        messages: List of previous messages in the conversation
        retrieval_result: Dictionary containing retrieved context and sources
        company_id: Company identifier for contextual responses
        company_type: Company type for suggested questions
    Returns:
        Dictionary with has_answer, answer, sources, follow_ups, and metadata
    """

    # Handle greetings
    if is_greeting(query):
        logger.info("Greeting detected, returning contextual response")
        return greeting_response(company_id, company_type)

    # Handle capability queries
    if is_capability_query(query):
        logger.info("Capability query detected, returning capability response")
        return capability_response(company_id, company_type)

    context = retrieval_result.get("context", "")
    sources = retrieval_result.get("sources", [])

    # Handle out-of-scope queries
    if not has_sufficient_context(context):
        logger.warning("Insufficient context for query")
        return out_of_scope_response(company_id, company_type, sources)

    logger.debug("Sufficient context available, building prompt")
    prompt = build_prompt(query, messages, context)

    try:
        logger.debug("Invoking LLM for answer generation")
        response = llm.invoke(prompt)
        raw_answer = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        answer, follow_ups = parse_follow_ups(raw_answer)
        logger.info("Answer generated successfully with %d follow-ups", len(follow_ups))
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return {
            "has_answer": False,
            "answer": f"Generation failed: {e}",
            "sources": sources,
            "follow_ups": [],
            "original_query": query,
        }

    return {
        "has_answer": True,
        "answer": answer,
        "sources": sources,
        "follow_ups": follow_ups,
        "original_query": query,
    }
