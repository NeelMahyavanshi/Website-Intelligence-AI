# ============================================================
# pipeline/store.py
# ============================================================
import os
from typing import Any
import chromadb
import hashlib
from chromadb import (
    Schema, VectorIndexConfig, SparseVectorIndexConfig,
    K, Knn, Rrf, Search, FtsIndexConfig,
)
from chromadb.utils.embedding_functions import (
    GoogleGeminiEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
import re
from chromadb.api.types import Where
from dotenv import load_dotenv
from pipeline.crawler import crawl_url
from pipeline.chunker import process_record
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)  

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
    """
    Agent picks a preset based on the company's website type.
    - space: distance metric for dense vectors
    - model: Gemini embedding model
    - dense_weight / sparse_weight: RRF fusion weights
    - task_type: Gemini task hint
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
    return hashlib.md5(content.encode()).hexdigest()

def chunk_id(url: str, text: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()
    return f"{url}#{h}"

def flatten_meta(meta: dict) -> dict:
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
    """Ensure the company_id matches Chroma's collection-name rules."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name or "")[:200]
    return cleaned or "default"

_collection_cache = {}

def get_collection(
    company_id: str,
    collection_name: str,
    company_type: str,
):
    cache_key = f"{company_id}_{collection_name}"

    if cache_key not in _collection_cache:

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
        
    return collection

# ============================================================
# SAFE UPSERT
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def safe_upsert(collection, batch):

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


# ============================================================
# INGEST WEBSITE
# ============================================================

async def ingest(
    start_url: str,
    company_id: str,
    company_type: str,
    collection_name: str = "web_docs"
) -> int:

    

    print("=" * 60)
    print("STARTING INGEST")
    print("=" * 60)

    # Crawl
    pages = await crawl_url(start_url)

    # Remove blank pages
    pages = [
        p for p in pages
        if p.get("content", "").strip()
    ]

    print("Pages found:", len(pages))

    collection = get_collection(
        company_id,
        collection_name,
        company_type
    )

    # Fetch existing content hashes to skip unchanged pages
    existing_hashes: set[str] = set()
    try:
        got = collection.get(include=["metadatas"], limit=10000)
        for m in (got.get("metadatas") or []):
            h = m.get("content_hash")
            if h:
                existing_hashes.add(h)
    except Exception as e:
        print("Could not read existing hashes:", e)


    all_chunks = []

    # Process each page
    for page in pages:

        try:
            h = page_hash(page["content"])
            if h in existing_hashes:
                print("Skipping unchanged page:", page["url"])
                continue

            page["metadata"]["content_hash"] = h

            chunks = process_record(page)

            all_chunks.extend(chunks)

            print("Processed:", page["url"])

        except Exception as e:
            print("Failed page:", page["url"], e)
            continue

    if not all_chunks:
        print("No chunks generated.")
        return 0

    print("Total chunks:", len(all_chunks))

    # Batch save
    BATCH_SIZE = 250

    for i in range(0, len(all_chunks), BATCH_SIZE):

        batch = all_chunks[i:i + BATCH_SIZE]

        safe_upsert(collection, batch)

        print(f"Saved batch {i} -> {i + len(batch)}")

    print("INGEST COMPLETE")

    return len(all_chunks)


# ============================================================
# BASIC HYBRID SEARCH
# ============================================================

def hybrid_query(
    company_id: str,
    company_type: str,
    query: str,
    k: int = 5,
    where: dict | Where | None = None,
    collection_name: str = "web_docs"
):

    try:

        cfg = build_company_config(company_type)

        collection = get_collection(
            company_id,
            collection_name,
            company_type
        )

        rank = Rrf(
            ranks=[

                Knn(
                    query=query,
                    return_rank=True
                ),

                Knn(
                    query=query,
                    key="sparse_embedding",
                    return_rank=True
                )
            ],

            weights=[
                cfg["dense_weight"],
                cfg["sparse_weight"]
            ],

            normalize=True
        )

        search = (
            Search()
            .rank(rank)
            .limit(k)
            .select(
                K.DOCUMENT,
                K.SCORE,
                K.METADATA
            )
        )

        if where is not None: 
            search = search.where(where)

        results = collection.search(search)

        return results

    except Exception as e:
        print("Search failed:", e)
        return []
