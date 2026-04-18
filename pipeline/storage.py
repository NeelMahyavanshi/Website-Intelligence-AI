import chromadb
from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import (
    GoogleGeminiEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from crawler import crawl_url

# ---- 1. Agent-configurable settings (placeholder) ----
def build_company_config(company_type: str) -> dict:
    """AI agent fills this in based on company type."""
    presets = {
        "tech_docs":   {"space": "cosine", "model": "gemini-embedding-001"},
        "support":     {"space": "cosine", "model": "gemini-embedding-001"},
        "ecommerce":   {"space": "l2",     "model": "gemini-embedding-001"},
        "default":     {"space": "cosine", "model": "gemini-embedding-001"},
    }
    return presets.get(company_type, presets["default"])


# ---- 2. Create (or fetch) a hybrid collection for a company ----

def get_collection(company_id:str, collection_name : str, company_type:str):

    cfg = build_company_config(company_type)

    client = chromadb.CloudClient(
        tenant="website_intelligence",  # your organization name in ChromaCloud
        database=company_id,         # each company = its own namespace
        api_key=os.getenv("CHROMADB_API_KEY"),
    )

    gemini_ef = GoogleGeminiEmbeddingFunction(
        model_name=cfg["model"],
        task_type="RETRIEVAL_DOCUMENT",
    )

    schema = Schema()

    schema.create_index(config=VectorIndexConfig(
        space=cfg["space"], embedding_function=gemini_ef
    ))

    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=ChromaCloudSpladeEmbeddingFunction(),
        ),
        key="sparse_embedding",

    )    
    
    return client.get_or_create_collection(name=collection_name, schema=schema)


def flatten_meta(m:dict) -> dict:
    """Chroma metadata values must be str/int/float/bool — flatten/drop None."""
    out = {}
    for k, v in (m or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

async def ingest(start_url, company_id, company_type, collection_name="web_docs"):
    pages = await crawl_url(start_url)
    pages = [p for p in pages if p["content"].strip()]   # skip empty

    collection = get_collection(company_id, collection_name, company_type)

    collection.add(
        ids=[f"{company_id}-{i}" for i in range(len(pages))],
        documents=[p["content"] for p in pages],
        metadatas=[
            {
                "url": p["url"],
                "timestamp": p["timestamp"],
                **flatten_meta(p["metadata"]),
            }
            for p in pages
        ],
    )