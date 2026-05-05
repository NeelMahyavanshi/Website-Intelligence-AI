# ============================================================
# pipeline/retriever.py
# ============================================================

from pydantic import BaseModel
from pipeline.store import get_collection, build_company_config
from llm_model import llm
from sentence_transformers import CrossEncoder
import hashlib
from chromadb.api.types import Where
from chromadb import K, Knn, Rrf, Search
import asyncio
from concurrent.futures import ThreadPoolExecutor
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv(override=True)

logger = get_logger("RETRIEVER")

# ============================================================
# QUERY REWRITE
# ============================================================


class RewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str

def rewrite_query(user_query:str) -> str: 
    """
    Rewrites user query to better match the embedding space.
    - For example, "How do I reset my password?" might be rewritten to "password reset instructions"
    - This can be done using a simple LLM prompt or regex-based rules.
    """
    
    prompt = f"""
        Rewrite this query for semantic search.

        Rules:
        - keep meaning SAME
        - DO NOT list keywords
        - DO NOT use "|" or commas
        - output ONE natural sentence

        Query: {user_query}

        Return:
        {{
        "original_query": "...",
        "rewritten_query": "..."
        }}
    """
    
    rewritten_query = llm.with_structured_output(RewrittenQuery)
    response = rewritten_query.invoke(prompt)
    return  response.original_query, response.rewritten_query
    
    
# ============================================================
# FILTER BUILDING
# ============================================================

class FilterConfig(BaseModel): 
    apply_filter: bool
    filter: dict | None
    

ALLOWED_FILTERS = ["content_type", "source_url"]

def safe_filter(f):
    if not f:
        return None
    filtered = {k: v for k, v in f.items() if k in ALLOWED_FILTERS}
    return filtered if filtered else None

def build_filter(query) -> dict | None:
    """
    Extracts structured filters from the query if present.
    - For example, "Show me articles about billing issues" might extract a filter {"topic": "billing issues"}
    - This can be done using regex or an LLM prompt to identify key-value pairs in the query.
    """
    
    prompt = f"""
    Extract structured filters from this search query if present. 
    If no filters are needed, return an empty object.        
    Query:
    {query}
    Return below JSON object with keys as filter names and values as filter values.
    {
        {
            "apply_filter": bool,
            "filter": dict | None
        }
    }


    """
    filter_extractor = llm.with_structured_output(FilterConfig)
    response = filter_extractor.invoke(prompt)
    return safe_filter(response.filter) if response.apply_filter else None


# ============================================================
# HYBRID QUERY
# ============================================================

def remove_duplicates(results: list[dict]) -> list[dict]:
    """
    Removes duplicate results based on document ID or content similarity.
    - This can be done by keeping track of seen document IDs or using a similarity threshold to filter out near-duplicates.
    """
    seen = set()
    unique = []

    for r in results:
        h = hashlib.md5((r["metadata"].get("source_url", "") + r["text"][:200]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(r)

    return unique


# ============================================================
# RERANKING
# ============================================================

# Reranker model - load once at module level
_rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
logger.debug("Reranker model loaded")

def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Reranks results using a more sophisticated model or additional features.
    - For example, you could use an LLM to score the relevance of each result to the query and sort them accordingly.
    - This can help improve the quality of the top results returned to the user.
    """

    scores = _rerank_model.predict([(query, r["text"]) for r in results])
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

# ============================================================
# CONFIDENCE FILTERING 
# ============================================================

def filter_confidence(results: list[dict], threshold: float = 0.02) -> list[dict]:
    """
    Filters out results that are below a certain confidence threshold.
    - This can be based on the similarity score from the hybrid query or an additional relevance score from a reranking step.
    - Setting an appropriate threshold can help ensure that only relevant results are returned to the user.
    """
    
    return [r for r in results if r.get("rerank_score", 0) >= threshold]


# ============================================================
# CONTEXT FORMATTING
# ============================================================

def format_context(results: list[dict]) -> tuple[str, list[str]]:
    passages = []
    sources = []
    
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("source_url", "unknown")
        section = meta.get("section_title", "")
        
        passages.append(
            f"[Source {i}] {source}\n"
            f"{section}\n"
            f"---\n"
            f"{r['text']}"
        )
        sources.append(source)
    
    return "\n\n".join(passages), sources


# ============================================================
# PARSE RAW RESULTS
# ============================================================

def parse_results(raw) -> list[dict]:
    if not raw:
        return []
    # Search API: list of rows
    if isinstance(raw, list):
        rows = raw[0] if raw and isinstance(raw[0], list) else raw
        return [{
            "id": r.get("id"),
            "text": r.get("document") or r.get("#document"),
            "metadata": r.get("metadata") or r.get("#metadata") or {},
            "score": float(r.get("score") or r.get("#score") or 0.0),
        } for r in rows]
    # Legacy fallback
    if raw.get("ids"):
        ids = raw["ids"][0]; docs = raw["documents"][0]
        metas = raw["metadatas"][0]; scores = raw["scores"][0]
        return [{"id": i, "text": d, "metadata": m, "score": float(s)}
                for i, d, m, s in zip(ids, docs, metas, scores)]
    return []


# ============================================================
# BASIC HYBRID SEARCH
# ============================================================

def hybrid_query(
    company_id: str,
    company_type: str,
    query_dense: str,
    query_sparse: str,
    k: int = 5,
    where: dict | Where | None = None,
    collection_name: str = "web_docs",
):
    """Execute hybrid search combining dense and sparse embeddings.
    
    Args:
        company_id: Company identifier
        company_type: Type of company (for config selection)
        query_dense: Query for dense semantic search
        query_sparse: Query for sparse keyword search
        k: Number of results to return
        where: Optional filter constraints
        collection_name: Name of the collection to search
        
    Returns:
        Raw search results from the collection
    """
    try:
        cfg = build_company_config(company_type)
        collection = get_collection(company_id, collection_name, company_type)
        logger.debug("Searching with k=%d, dense_weight=%.1f", k, cfg["dense_weight"])

        rank = Rrf(
            ranks=[
                Knn(query=query_dense, return_rank=True),
                Knn(query=query_sparse, key="sparse_embedding", return_rank=True),
            ],
            weights=[cfg["dense_weight"], cfg["sparse_weight"]],
            normalize=True,
        )

        search = (
            Search()
            .rank(rank)
            .limit(k)
            .select(K.DOCUMENT, K.SCORE, K.METADATA)
        )

        if where and isinstance(where, dict) and len(where) > 0:
            search = search.where(where)
            logger.debug("Applied filter: %s", list(where.keys()))

        return collection.search(search)

    except Exception as e:
        logger.error("Hybrid search failed: %s", e, exc_info=True)
        return {"rows": []}
    


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve(query: str, company_id: str, company_type: str, k: int = 10):
    """Main retrieval function that orchestrates query rewriting, filtering, hybrid search, reranking, and result formatting.
    
    Args:
        query: User's search query
        company_id: Company identifier
        company_type: Type of company (for config selection)
        k: Number of top results to return
        
    Returns:
        Dictionary with context, sources, chunks, and metadata
    """
    logger.info("Retrieval start for: %s (k=%d)", company_id, k)

    # Parallel execution of query rewriting and filter building
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_rewrite = executor.submit(rewrite_query, query)
        future_filter = executor.submit(build_filter, query)
        
        original_query, rewritten = future_rewrite.result()
        filters = future_filter.result()

    logger.debug("Query rewritten: %s", rewritten[:60] if rewritten else rewritten)

    # Retrieve a wider pool, rerank narrows it down
    raw_results = hybrid_query(
        company_id=company_id,
        company_type=company_type,
        query_dense=rewritten,        
        query_sparse=original_query,  
        k=max(k * 4, 30),
        where=filters,
    )

    parsed = parse_results(raw_results)
    logger.debug("Retrieved %d results", len(parsed))

    unique = remove_duplicates(parsed)
    logger.debug("After deduplication: %d results", len(unique))

    if not unique:
        logger.warning("No results found for query")
        return {"context": "", "sources": [], "chunks": [],
                "rewritten_query": rewritten, "metadata": []}

    # Rerank with the ORIGINAL user query (more faithful to intent)
    try:
        reranked = rerank(original_query, unique)
        logger.debug("Reranking complete")
    except Exception as e:
        logger.warning("Reranking failed: %s, using unranked results", e)
        reranked = unique

    # Take top-k after reranking, then apply confidence
    top = reranked[:k]
    confident = filter_confidence(top)
    if not confident:
        logger.debug("No results passed confidence threshold, using top 3")
        confident = top[:3]

    context, sources = format_context(confident)
    logger.info("Retrieval complete: %d results, confidence filtered", len(confident))

    return {
        "context": context,
        "sources": sources,
        "chunks": [r["text"] for r in confident],
        "rewritten_query": rewritten,
        "original_query": original_query,
        "metadata": [r["metadata"] for r in confident],
    }