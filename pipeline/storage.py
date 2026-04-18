import chromadb
from chromadb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from chromadb.utils.embedding_functions import (
    GoogleGeminiEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
import os
from dotenv import load_dotenv
load_dotenv(override=True)

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

def get_or_create_collection(company_id:str, collection_name : str, company_type:str):

    cfg = build_company_config(company_type)

    client = chromadb.CloudClient(
        tenant="website_intelligence",  # your organization name in ChromaCloud
        database=company_id,         # each company = its own namespace
        api_key="your-api-key",
    )