# retriever.py

def rewrite_query(query) -> str :
    """
    Rewrites the user query to improve retrieval performance.
    This can include:
    - Adding context or clarifying intent
    - Expanding with synonyms or related terms
    - Simplifying complex queries
    """
    # Placeholder for actual rewriting logic, e.g. using an LLM
    rewritten = query.strip()
    if len(rewritten) < 10:
        rewritten += " documentation"
    return rewritten


def build_filter(query) -> dict | None:
    """
    Builds a filter dictionary for retrieval based on query analysis.
    This can include:
    - Extracting entities or keywords from the query
    - Mapping to metadata fields for filtering
    """
    # Placeholder for actual filter building logic, e.g. using an LLM
    if "pricing" in query.lower():
        return {"content_type": "pricing"}
    elif "api" in query.lower() or "docs" in query.lower():
        return {"content_type": "docs"}
    else:
        return None
     
def remove_duplicates(results) -> list:
    """
    Removes duplicate results based on URL or content hash.
    """
    seen_urls = set()
    unique_results = []
    for r in results:
        url = r.get("metadata", {}).get("source_url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    return unique_results

def rerank(query, results) -> list:
    """
    Reranks the results based on relevance to the query.
    """
    # Placeholder for actual reranking logic, e.g. using an LLM
    return results

def filter_confidence(results) -> list:
    """
    Filters results based on confidence scores.
    """
    # Placeholder for actual filtering logic
    return results

def format_context(results) -> str:
    """
    Formats the retrieved results into a context string for LLM input.
    """
    # Placeholder for actual formatting logic
    return "\n".join(r.get("text", "") for r in results)

def retrieve(query, company_id, company_type) -> dict:
    """
    Main retrieval function that orchestrates the retrieval pipeline.
    """
    print(f"[RETRIEVER] Starting retrieval for query: '{query}'")
    
    # Step 1: Rewrite the query
    rewritten_query = rewrite_query(query)
    print(f"[RETRIEVER] Rewritten query: '{rewritten_query}'")
    
    # Step 2: Build filter based on query
    filter_ = build_filter(rewritten_query)
    print(f"[RETRIEVER] Built filter: {filter_}")
    
    # Step 3: Perform initial retrieval
    results = hybrid_query(
        company_id=company_id,
        company_type=company_type,
        query=rewritten_query,
        k=10,
        where=filter_
    )
    print(f"[RETRIEVER] Initial results count: {len(results)}")
    
    # Step 4: Remove duplicates
    unique_results = remove_duplicates(results)
    print(f"[RETRIEVER] Unique results count after deduplication: {len(unique_results)}")
    
    # Step 5: Rerank results
    reranked_results = rerank(rewritten_query, unique_results)
    
    # Step 6: Filter by confidence
    filtered_results = filter_confidence(reranked_results)
    
    # Step 7: Format context for LLM input
    context = format_context(filtered_results)
    
    return {
        "results": filtered_results,
        "context": context
    }