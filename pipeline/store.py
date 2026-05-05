# ============================================================
# pipeline/store.py
# ============================================================
import os
from typing import Any
import chromadb
import hashlib
from chromadb import (
    Schema, VectorIndexConfig, SparseVectorIndexConfig,
    K
)
from chromadb.utils.embedding_functions import (
    GoogleGeminiEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
import re
from dotenv import load_dotenv
from pipeline.crawler import crawl_url
from pipeline.chunker import process_record
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import get_logger

load_dotenv(override=True)

logger = get_logger("STORE")

# ============================================================
# GLOBAL EMBEDDING FUNCTION
# ============================================================

gemini_ef = GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    dimension=768
)

# ============================================================
# COMPANY CONFIG
# ============================================================

def build_company_config(company_type: str) -> dict:
    """Retrieves the vector storage config for a given company type.
    
    Different company types benefit from different embedding spaces and RRF weights:
    - space: distance metric for dense vectors (cosine/l2)
    - model: Gemini embedding model for dense vectors
    - dense_weight/sparse_weight: Relative importance in RRF fusion
    - task_type: Gemini task hint for better embeddings
    
    Args:
        company_type: Type of company (tech_docs, support, ecommerce, blog, or default)
        
    Returns:
        Dictionary with space, model, dense_weight, sparse_weight, and task_type
    """
    presets = {
        "tech_docs":  {"space": "cosine", "model": "gemini-embedding-001",
                       "dense_weight": 0.6, "sparse_weight": 0.4,
                       "task_type": "RETRIEVAL_DOCUMENT"},
        "support":    {"space": "cosine", "model": "gemini-embedding-001",
                       "dense_weight": 0.7, "sparse_weight": 0.3,
                       "task_type": "RETRIEVAL_DOCUMENT"},
        "ecommerce":  {"space": "l2",     "model": "gemini-embedding-001",
                       "dense_weight": 0.5, "sparse_weight": 0.5,
                       "task_type": "RETRIEVAL_DOCUMENT"},
        "blog":       {"space": "cosine", "model": "gemini-embedding-001",
                       "dense_weight": 0.8, "sparse_weight": 0.2,
                       "task_type": "RETRIEVAL_DOCUMENT"},
        "default":    {"space": "cosine", "model": "gemini-embedding-001",
                       "dense_weight": 0.7, "sparse_weight": 0.3,
                       "task_type": "RETRIEVAL_DOCUMENT"},
    }
    return presets.get(company_type, presets["default"])


# ============================================================
# HELPERS
# ============================================================


def page_hash(content: str) -> str:
    """Computes MD5 hash of page content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()

def chunk_id(url: str, text: str) -> str:
    """Generates a unique chunk ID from URL and text hash."""
    h = hashlib.md5(text.encode()).hexdigest()
    return f"{url}#{h}"

def flatten_meta(meta: dict) -> dict:
    """Flattens metadata dict to only include Chroma-compatible types.
    
    Converts lists to comma-separated strings and serializes complex objects.
    Chroma requires metadata values to be str, int, float, or bool.
    """
    clean = {}

    for k, v in (meta or {}).items():

        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            clean[k] = v

        elif isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v)

        else:
            clean[k] = str(v)

    return clean


# ============================================================
# SCHEMA
# ============================================================

def build_schema(cfg: dict) -> Schema:
    """Builds Chroma schema with both dense and sparse vector indexes.
    
    Creates two indexes:
    1. Dense vectors via Gemini embeddings for semantic search
    2. Sparse vectors via SPLADE for keyword matching
    
    Uses RRF fusion in retriever to combine both signals.
    
    Args:
        cfg: Config dict with 'space' key (cosine/l2)
        
    Returns:
        Chroma Schema configured with dual indexes
    """
    schema = Schema()

    # Dense semantic search
    schema.create_index(
        config=VectorIndexConfig(
            space=cfg["space"],
            embedding_function=gemini_ef
        )
    )

    # Sparse keyword search
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=ChromaCloudSpladeEmbeddingFunction()
        ),
        key="sparse_embedding"
    )

    return schema

# ============================================================
# COLLECTION
# ============================================================

def _safe_id(name: str) -> str:
    """Sanitizes company_id to match Chroma's collection-name rules.
    
    Chroma allows only alphanumeric, dots, underscores, hyphens (max 200 chars).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name or "")[:200]
    return cleaned or "default"

_collection_cache = {}

def get_collection(
    company_id: str,
    collection_name: str,
    company_type: str,
) -> Any:
    """Gets or creates a Chroma collection with caching.
    
    Handles collection initialization with dual indexes (dense + sparse).
    Collections are cached in memory to avoid repeated cloud API calls.
    
    Args:
        company_id: Unique company identifier
        collection_name: Name of the collection (e.g., 'web_docs')
        company_type: Type determines embedding space and RRF weights
        
    Returns:
        Chroma collection object ready for upsert/query operations
    """
    cache_key = f"{company_id}_{collection_name}"

    if cache_key not in _collection_cache:
        logger.debug("Creating collection for %s/%s (type=%s)", company_id, collection_name, company_type)

        cfg = build_company_config(company_type)

        client = chromadb.CloudClient(
            tenant=os.getenv("CHROMADB_TENANT"),
            database="main",
            api_key=os.getenv("CHROMADB_API_KEY"),
        )

        full_collection_name = f"{_safe_id(company_id)}_{collection_name}"

        collection = client.get_or_create_collection(
            name=full_collection_name,
            schema=build_schema(cfg),
            metadata={
                "company_id": company_id,
                "company_type": company_type,
            },
        )

        _collection_cache[cache_key] = collection
        logger.info("Collection ready: %s (dense_weight=%.1f, sparse_weight=%.1f)", full_collection_name, cfg["dense_weight"], cfg["sparse_weight"])
        
    return _collection_cache[cache_key]

# ============================================================
# SAFE UPSERT
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def safe_upsert(collection, batch) -> None:
    """Safely upserts a batch of chunks to the collection with retry logic.
    
    Handles transient errors via exponential backoff (up to 3 attempts).
    Generates deterministic chunk IDs from URL + text hash for deduplication.
    Flattens metadata to Chroma-compatible types.
    
    Args:
        collection: Chroma collection object
        batch: List of dicts with 'text' and 'metadata' keys
        
    Raises:
        Exception: If upsert fails after 3 retries
    """
    ids = [
        chunk_id(
            c["metadata"]["source_url"],
            c["text"]
        )
        for c in batch
    ]

    docs = [
        c["text"]
        for c in batch
    ]

    metas = [
        flatten_meta(c["metadata"])
        for c in batch
    ]

    collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas
    )
    logger.debug("Upserted batch of %d chunks", len(batch))


# ============================================================
# INGEST WEBSITE
# ============================================================

async def ingest(
    start_url: str,
    company_id: str,
    company_type: str,
    collection_name: str = "web_docs"
) -> int:
    """Ingests a website into the vector store via full pipeline.
    
    Pipeline:
    1. Crawl website using intelligent BFS strategy
    2. Remove blank pages
    3. Chunk each page using LLM-based intelligent chunking
    4. Deduplicate against existing collection (by content hash)
    5. Batch upsert to Chroma with automatic retries
    
    Args:
        start_url: Root URL to crawl
        company_id: Company identifier for collection naming
        company_type: Type affects embedding space and RRF weights
        collection_name: Defaults to 'web_docs'
        
    Returns:
        Total number of chunks ingested
    """
    logger.info("Starting ingestion pipeline for %s (company_type=%s)", start_url, company_type)

    # Crawl website
    pages = await crawl_url(start_url)

    # Remove blank pages
    pages = [
        p for p in pages
        if p.get("content", "").strip()
    ]

    logger.info("Crawled %d pages from %s", len(pages), start_url)

    collection = get_collection(
        company_id,
        collection_name,
        company_type
    )

    # Fetch existing content hashes to skip unchanged pages
    existing_hashes: set[str] = set()
    try:
        got = collection.get(include=["metadatas"], limit=300)
        for m in (got.get("metadatas") or []):
            h = m.get("content_hash")
            if h:
                existing_hashes.add(h)
        logger.debug("Found %d existing content hashes", len(existing_hashes))
    except Exception as e:
        logger.warning("Could not read existing hashes: %s", str(e))


    all_chunks = []

    # Process each page: intelligent chunking with deduplication
    for page in pages:

        try:
            h = page_hash(page["content"])
            if h in existing_hashes:
                logger.debug("Skipping unchanged page: %s", page["url"])
                continue

            page["metadata"]["content_hash"] = h

            # LLM-based intelligent chunking with validation
            chunks = process_record(page)

            for c in chunks:
                c.setdefault("metadata", {})
                c["metadata"]["content_hash"] = h
                # Preserve source URL for retrieval result formatting
                c["metadata"].setdefault("source_url", page.get("url"))


            all_chunks.extend(chunks)

            logger.debug("Processed %d chunks from: %s", len(chunks), page["url"])

        except Exception as e:
            logger.error("Failed to process page %s: %s", page["url"], str(e), exc_info=True)
            continue

    if not all_chunks:
        logger.warning("No chunks generated from any pages")
        return 0

    logger.info("Generated %d total chunks from %d pages", len(all_chunks), len(pages))

    # Batch save to Chroma with automatic retries
    BATCH_SIZE = 250

    for i in range(0, len(all_chunks), BATCH_SIZE):

        batch = all_chunks[i:i + BATCH_SIZE]

        safe_upsert(collection, batch)

        logger.info("Saved batch %d/%d (%d chunks)", i // BATCH_SIZE + 1, (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE, len(batch))

    logger.info("Ingest complete: %d chunks successfully stored", len(all_chunks))

    return len(all_chunks)