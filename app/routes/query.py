from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pipeline.retriever import (
    retrieve, rewrite_query, build_filter,
    hybrid_query, parse_results, remove_duplicates, rerank, format_context
)
from pipeline.generator import (
    generate, is_greeting, is_capability_query,
    greeting_response, capability_response, out_of_scope_response
)
from utils.helpers import extract_company_id
from utils.database import db
import asyncio
import json
import time
from utils.logger import get_logger

logger = get_logger("QUERY_ROUTE")

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    url:str
    messages: list[dict] | None = None

@router.post("/stream")
async def query_stream(request: QueryRequest):
    """SSE endpoint that streams retrieval pipeline stages + final answer."""

    async def event_stream():
        def emit(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"
    
        try:
            company_id = extract_company_id(request.url)
            query = request.query
            loop = asyncio.get_event_loop()

            # Short-circuit greetings/capability — no retrieval needed
            job = db.table("ingest_jobs")\
                .select("company_type")\
                .eq("company_id", company_id)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            company_type = job.data[0]["company_type"] if job.data else "default"

            if is_greeting(query):
                yield emit({"type": "answer", **greeting_response(company_id, company_type), "chunks": []})
                return
            if is_capability_query(query):
                yield emit({"type": "answer", **capability_response(company_id, company_type), "chunks": []})
                return

            # ── Stage 1: Rewrite query ──────────────────────────────────
            yield emit({"type": "stage_start", "stage": "rewriting", "label": "Rewriting query"})
            t0 = time.time()
            original_query, rewritten = await loop.run_in_executor(None, rewrite_query, query)
            yield emit({"type": "stage_done", "stage": "rewriting", "ms": int((time.time() - t0) * 1000)})

            # ── Stage 2: Build filters ──────────────────────────────────
            yield emit({"type": "stage_start", "stage": "filters", "label": "Building filters"})
            t0 = time.time()
            filters = await loop.run_in_executor(None, build_filter, query)
            yield emit({"type": "stage_done", "stage": "filters", "ms": int((time.time() - t0) * 1000)})

            # ── Stage 3: Hybrid retrieval ───────────────────────────────
            yield emit({"type": "stage_start", "stage": "retrieval", "label": "Hybrid retrieval"})
            t0 = time.time()
            k = 15
            raw = await loop.run_in_executor(
                None, hybrid_query,
                company_id, company_type, rewritten, original_query, max(k * 4, 30), filters
            )
            parsed = parse_results(raw)
            unique = remove_duplicates(parsed)
            yield emit({"type": "stage_done", "stage": "retrieval",
                        "ms": int((time.time() - t0) * 1000), "count": len(unique)})

            if not unique:
                resp = out_of_scope_response(company_id, company_type, [])
                yield emit({"type": "answer", **resp, "chunks": []})
                return

            # ── Stage 4: Rerank ─────────────────────────────────────────
            yield emit({"type": "stage_start", "stage": "reranking", "label": "Reranking chunks"})
            t0 = time.time()
            try:
                reranked = await loop.run_in_executor(None, rerank, original_query, unique)
            except Exception:
                reranked = unique
            top = reranked[:k]
            context, sources = format_context(top)
            yield emit({"type": "stage_done", "stage": "reranking",
                        "ms": int((time.time() - t0) * 1000), "count": len(top)})

            # ── Stage 5: Generate answer ────────────────────────────────
            yield emit({"type": "stage_start", "stage": "generating", "label": "Building answer"})
            t0 = time.time()
            retrieval_result = {
                "context": context, "sources": sources,
                "chunks": [r["text"] for r in top],
                "company_type": company_type,
                "metadata": [r["metadata"] for r in top],
            }
            generation = await loop.run_in_executor(
                None, generate, query, request.messages, retrieval_result, company_id, company_type
            )
            yield emit({"type": "stage_done", "stage": "generating",
                        "ms": int((time.time() - t0) * 1000)})

            # Chunk metadata for source drawer (top 5)
            chunks_meta = [
                {
                    "text": r["text"],
                    "url": r["metadata"].get("source_url", ""),
                    "section": r["metadata"].get("section_title", ""),
                    "score": round(float(r.get("rerank_score", 0)), 3),
                }
                for r in top[:5]
            ]

            yield emit({"type": "answer", **generation, "chunks": chunks_meta})

        except Exception as e:
            logger.error("Stream query failed: %s", e, exc_info=True)
            yield emit({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/")
async def query_endpoint(request: QueryRequest):

    """
    Handles incoming query requests. It takes a user query and company information, retrieves relevant data from the vector database,
    and generates a response using the LLM.
    """

    company_id = None
    try:
        loop = asyncio.get_event_loop()
        company_id = extract_company_id(request.url)
        retrieval = await loop.run_in_executor(
            None, retrieve, request.query, company_id
        )
        company_type = retrieval.get("company_type", "default")
        generation = await loop.run_in_executor(
            None, generate, request.query, request.messages, retrieval, company_id, company_type
        )
        return generation
    except Exception as e:
        if isinstance(e, ValueError) and "No collection found" in str(e):
            logger.warning("No data found for company ID: %s", company_id)
            raise HTTPException(status_code=404, detail="No data found for this company")
        logger.error("Error processing query: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    


    