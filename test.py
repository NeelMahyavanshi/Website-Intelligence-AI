# ============================================================
# test.py
# FULL LOCAL TEST FOR store.py WITHOUT REAL CRAWLING
# ============================================================
import os
import re
import asyncio
import chromadb
from chromadb import K
from dotenv import load_dotenv

from pipeline.chunker import process_record
from pipeline.store import (
    get_collection,
    safe_upsert,
    hybrid_query,
)
from pipeline.store import _safe_id
load_dotenv(override=True)


# ============================================================
# STEP 0: ENV CHECK
# ============================================================

REQUIRED_ENV = ("CHROMADB_API_KEY", "CHROMADB_TENANT", "GEMINI_API_KEY")
for var in REQUIRED_ENV:
    assert os.getenv(var), f"Missing env var: {var}"


# ============================================================
# STEP 1: FAKE CRAWLER OUTPUT
# ============================================================

sample_page = {
    "url": "https://demo-company.com/docs/authentication",
    "content": """
# Authentication Guide

Welcome to Demo Company API.

Use POST /login to authenticate users.

Headers:
Authorization: Bearer <token>

Request Body:
{
  "email": "user@example.com",
  "password": "secret"
}

Response:
{
  "access_token": "jwt_token_here",
  "expires_in": 3600
}

Error Codes:
401 Unauthorized
403 Forbidden
429 Too Many Requests

Security Notes:
- Always use HTTPS
- Tokens expire after 1 hour
- Refresh tokens supported

Python Example:

import requests

response = requests.post(
    "https://api.demo-company.com/login",
    json={
        "email": "user@example.com",
        "password": "secret"
    }
)
""",
    "metadata": {
        "title": "Authentication Guide",
        "description": "How login works for Demo Company API",
        "depth": 1,
        "parent_url": "https://demo-company.com/docs",
    },
    "timestamp": "2026-01-01T10:00:00",
}


# ============================================================
# STEP 2: CONFIG
# ============================================================

company_id = "demo_company"
company_type = "tech_docs"
collection_name = "web_docs"


def _safe(name: str) -> str:
    """Match the sanitization your store.get_collection expects."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)[:200]


FULL_COLLECTION_NAME = f"{_safe_id(company_id)}_{collection_name}"


# ============================================================
# STEP 3: HELPERS
# ============================================================

def reset_collection():
    """Delete the collection so tests start clean."""
    client = chromadb.CloudClient(
        tenant=os.getenv("CHROMADB_TENANT"),
        database="main",
        api_key=os.getenv("CHROMADB_API_KEY"),
    )
    try:
        client.delete_collection(FULL_COLLECTION_NAME)
        print(f"Deleted existing collection: {FULL_COLLECTION_NAME}")
    except Exception as e:
        print(f"No existing collection to delete ({e.__class__.__name__}: {e})")


def print_results(results):
    if not results:
        print("No results.")
        return

    # Handle raw dict response from ChromaDB
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("scores", [[]])[0]

    if not ids:
        print("No results.")
        return

    for i, (id_, doc, meta, score) in enumerate(zip(ids, docs, metas, scores), 1):
        print(f"\nResult {i}")
        print("  Score:   ", score)
        print("  URL:     ", meta.get("source_url"))
        print("  Section: ", meta.get("section_title"))
        print("  Snippet: ", doc[:200].replace("\n", " "))


# ============================================================
# STEP 4: MAIN TEST
# ============================================================

async def main():

    print("=" * 60)
    print("STEP 0 — RESET COLLECTION")
    print("=" * 60)
    reset_collection()

    print("\n" + "=" * 60)
    print("STEP 1 — CREATE COLLECTION")
    print("=" * 60)
    collection = get_collection(
        company_id=company_id,
        collection_name=collection_name,
        company_type=company_type,
    )
    print("Collection ready:", collection.name)

    print("\n" + "=" * 60)
    print("STEP 2 — CHUNK PAGE")
    print("=" * 60)
    chunks = process_record(sample_page)
    print("Chunks created:", len(chunks))
    for i, c in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print("  Text:    ", c["text"][:200].replace("\n", " "))
        print("  Metadata:", c["metadata"])

    if not chunks:
        print("No chunks produced — aborting.")
        return

    print("\n" + "=" * 60)
    print("STEP 3 — UPSERT TO CHROMA")
    print("=" * 60)
    BATCH_SIZE = 250
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        safe_upsert(collection, batch)
        print(f"  Saved batch {i} -> {i + len(batch)}")
    print("All chunks stored.")

    print("\n" + "=" * 60)
    print("STEP 4 — COUNT")
    print("=" * 60)
    print("Total records in collection:", collection.count())

    print("\n" + "=" * 60)
    print("STEP 5 — HYBRID QUERY (no filter)")
    print("=" * 60)
    results = hybrid_query(
        company_id=company_id,
        company_type=company_type,
        query="How do users login and what are auth errors?",
        k=5,
    )
    print_results(results)

    print("\n" + "=" * 60)
    print("STEP 6 — HYBRID QUERY (with metadata filter)")
    print("=" * 60)
    results = hybrid_query(
        company_id=company_id,
        company_type=company_type,
        query="token expiration security",
        k=3,
        where={"source_url": {"$contains": "demo-company.com"}}
    )
    print_results(results)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())