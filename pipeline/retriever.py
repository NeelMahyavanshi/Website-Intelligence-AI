# ============================================================
# pipeline/retriever.py
# ============================================================

from pydantic import BaseModel
from pipeline.store import get_collection, build_company_config
from llm_model import llm
import hashlib
from chromadb.api.types import Where
from chromadb import K, Knn, Rrf, Search
from utils.database import db
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor
from utils.logger import get_logger
from dotenv import load_dotenv
import os
import cohere

load_dotenv(override=True)

logger = get_logger("RETRIEVER")

# ============================================================
# QUERY REWRITING
# ============================================================


class RewrittenQuery(BaseModel):
    original_query: str
    rewritten_query: str

@traceable(name="rewrite_query")
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
        - For example, "How do I reset my password?" might be rewritten to "password reset instructions"

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

@traceable(name="build_filters")
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
    {{
    "apply_filter": bool,
    "filter": dict | None
    }}
    """
    filter_extractor = llm.with_structured_output(FilterConfig)
    response = filter_extractor.invoke(prompt)
    return safe_filter(response.filter) if response.apply_filter else None


# ============================================================
# DEDUPLICATION
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

co = None

@traceable(name="rerank")
def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Reranks results using a more sophisticated model or additional features.
    - For example, you could use an LLM to score the relevance of each result to the query and sort them accordingly.
    - This can help improve the quality of the top results returned to the user.
    """
    # 1. get COHERE_API_KEY from env, if missing return results as-is

    if not os.getenv("COHERE_API_KEY"):
        logger.info("COHERE_API_KEY not set, skipping reranking")
        return results

    # 2. init cohere client (lazy, cache in module-level variable)
    global co
    if co is None:
        co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
    
    # 3. extract just the text from each result to send to Cohere

    text_results = [r["text"] for r in results]
    
    # 4. call cohere client.rerank() with model, query, documents, top_n
    try:
        response = co.rerank(
            model="rerank-v4.0-pro",
            query=query,
            documents=text_results,
            top_n=15
        )

        reranked_list = []
        for item in response.results:
            r = results[item.index].copy()
            r["rerank_score"] = float(item.relevance_score)
            reranked_list.append(r)

        return reranked_list
    
    except Exception as e:
        logger.warning("Cohere reranking failed: %s, using original order", e)
        return results

# ============================================================
# RESULT FORMATTING
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
# HYBRID SEARCH
# ============================================================

@traceable(name="hybrid_query")
def hybrid_query(
    company_id: str,
    company_type: str,
    query_dense: str,
    query_sparse: str,
    k: int = 15,
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
# RETRIEVAL ORCHESTRATION
# ============================================================
@traceable(name="retrieve")  
def retrieve(query: str, company_id: str, k: int = 15):
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
    
    job = db.table("ingest_jobs")\
        .select("company_type")\
        .eq("company_id", company_id)\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    
    company_type = job.data[0]["company_type"] if job.data else "default"
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
                "rewritten_query": rewritten, "metadata": [], "company_type": company_type}

    # Rerank with the ORIGINAL user query (more faithful to intent)
    try:
        reranked = rerank(original_query, unique)
        logger.debug("Reranking complete")
    except Exception as e:
        logger.warning("Reranking failed: %s, using unranked results", e)
        reranked = unique

    # Take top-k after reranking, then apply confidence
    # top = reranked[:k]
    # confident = filter_confidence(top)
    # if not confident:
    #     logger.debug("No results passed confidence threshold, using top 3")
    #     confident = confident or top[:3]

    top = reranked[:k]
    confident = top if top else top[:3]

    context, sources = format_context(confident)
    logger.info("Retrieval complete: %d results, confidence filtered", len(confident))

    return {
        "context": context,
        "sources": sources,
        "chunks": [r["text"] for r in confident],
        "rewritten_query": rewritten,
        "original_query": original_query,
        "metadata": [r["metadata"] for r in confident],
        "company_type": company_type,
    }