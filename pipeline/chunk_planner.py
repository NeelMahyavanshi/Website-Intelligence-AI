from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Literal
from pipeline.prompts.chunk_planner_prompt import chunk_planner_prompt as SYSTEM_PROMPT
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class ChunkPlan(BaseModel):
    page_type: Literal["docs", "product", "blog", "support", "faq", "pricing", "legal", "general"] = Field(..., description="Type of page: docs, product, ecommerce, blog, support, faq, pricing, legal, general")
    chunk_style: Literal["section_based", "semantic_flow", "feature_spec","qa_pairs","pricing_blocks","clause_exact", "thematic"] = Field(..., description="Preferred chunking style: 'thematic', 'section-based', 'semantic'")
    target_chunk_words: int = Field(...,ge=180, le=800, description="Ideal number of words per chunk, e.g. 300, between 180 - 800")
    preserve_code_blocks: bool = Field(..., description="Whether to preserve code blocks in the chunks")
    preserve_tables: bool = Field(..., description="Whether to preserve tables in the chunks")
    merge_short_sections: bool = Field(..., description="Whether to merge very short sections into adjacent chunks")
    metadata_focus: List[str] = Field(..., description="Which metadata fields to prioritize, e.g. ['page_title', 'section_title']"),
    notes: str = Field("",description="Any specific instructions for chunking this page")

def heuristic_router(record: dict) -> ChunkPlan:
    url = record.get("url", "unknown")
    title = record.get("metadata", {}).get("title", "").lower()

    if "/docs" in url or "/api/" in url or "/guides" in url or "/tutorials" in url or "/reference" in url or "documentation" in title:
        return ChunkPlan(
            page_type="docs",
            chunk_style="section_based",
            target_chunk_words=320,
            preserve_code_blocks=True,
            preserve_tables=True,
            merge_short_sections=True,
            metadata_focus=["title","section","entities"],
            notes="Preserve examples"
        )
    elif "pricing" in url or "plans" in url or "billing" in url or "pricing" in title:
        return ChunkPlan(
            page_type="pricing",
            chunk_style="pricing_blocks",
            target_chunk_words=200,
            preserve_code_blocks=False,
            preserve_tables=True,
            merge_short_sections=False,
            metadata_focus=["title", "price", "specs"],
            notes="Keep each pricing tier as one chunk, never split across tiers"
        )

    elif "/product" in url or "/features" in url or "buy" in url or "product" in title or "features" in title:
        return ChunkPlan(
            page_type="product",
            chunk_style="feature_spec",
            target_chunk_words=250,
            preserve_code_blocks=False,
            preserve_tables=True,
            merge_short_sections=False,
            metadata_focus=["title", "specs", "entities"],
            notes="Keep product specs and feature descriptions together"
        )
    elif "/blog" in url or "article" in url or "published" in url or "blog" in title:
        return ChunkPlan(
            page_type="blog",
            chunk_style="semantic_flow",
            target_chunk_words=400,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=True,
            metadata_focus=["title","summary","keywords"],
            notes="Preserve chronology"
        )
    
    elif "/faq" in url or "/help" in url or "faq" in title:
        return ChunkPlan(
            page_type="faq",
            chunk_style="qa_pairs",
            target_chunk_words=200,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=False,
            metadata_focus=["title", "summary"],
            notes="Keep each Q&A pair together, never split"
        )
    elif "/legal" in url or "/terms" in url or "/privacy" in url:
        return ChunkPlan(
            page_type="legal",
            chunk_style="clause_exact",
            target_chunk_words=250,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=False,
            metadata_focus=["title", "section"],
            notes="Preserve clause numbering"
        )
    else:
        return ChunkPlan(
            page_type="other",
            chunk_style="thematic",
            target_chunk_words=300,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=True,
            metadata_focus=["title","summary"],
            notes=""
        )


llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"), temperature=0)

llm_chunk_planner = llm.with_structured_output(ChunkPlan)

def llm_planner(record: dict) -> ChunkPlan:

    """
    Analyze the crawled record and create a chunking plan.

    """

    url = record.get("url", "unknown")
    raw_html = record.get("content", "")[:3000]
    crawled_metadata = record.get("metadata", {})
    ChunkPlan_config = record.get("crawl_config", {})  

    response = llm_chunk_planner.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"""
            URL: {url}

            TITLE: {crawled_metadata.get("title", "unknown")}

            DESCRIPTION: {crawled_metadata.get("description", "unknown")}

            CHUNK_PLAN_CONFIG: {ChunkPlan_config}

            CONTENT SAMPLE:
            {raw_html[:3000]} 
            """)
            ]
    )
    return response

def create_chunk_plan(record: dict) -> ChunkPlan:

    try:
        heuristic_router_plan = heuristic_router(record)
        if heuristic_router_plan.page_type != "other":
            print(f"Heuristic router assigned page_type={heuristic_router_plan.page_type} for URL: {record.get('url', 'unknown')}")
            return heuristic_router_plan
        else:
            print(f"Heuristic router could not confidently classify page, falling back to LLM planner for URL: {record.get('url', 'unknown')}")
            return llm_planner(record)
    except Exception as e:
        print(f"Chunk planning failed for URL: {record.get('url', 'unknown')}, error: {e}, using default chunk plan")
        return ChunkPlan(
            page_type="general",
            chunk_style="semantic",
            target_chunk_words=300,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=True,
            metadata_focus=["keywords"],
            notes="fallback"
        )