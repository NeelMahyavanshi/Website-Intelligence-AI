"""Website Intelligence AI - Chunk Planner Module

Determines optimal chunking strategy for crawled web pages using heuristic routing and LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Literal
from pipeline.prompts.chunk_planner_prompt import chunk_planner_prompt as SYSTEM_PROMPT
from llm_model import llm
from utils.logger import get_logger
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logger = get_logger("CHUNK_PLANNER")

# ============================================================
# CHUNK PLAN SCHEMA
# ============================================================

class ChunkPlan(BaseModel):
    """Schema for optimal chunking strategy determined for a page.
    
    Defines how a page should be split into chunks based on its content type.
    """
    page_type: Literal["docs", "product", "blog", "support", "faq", "pricing", "legal", "general"] = Field(..., description="Type of page: docs, product, ecommerce, blog, support, faq, pricing, legal, general")
    chunk_style: Literal["section_based", "semantic_flow", "feature_spec","qa_pairs","pricing_blocks","clause_exact", "thematic"] = Field(..., description="Preferred chunking style: 'thematic', 'section-based', 'semantic'")
    target_chunk_words: int = Field(...,ge=180, le=800, description="Ideal number of words per chunk, e.g. 300, between 180 - 800")
    preserve_code_blocks: bool = Field(..., description="Whether to preserve code blocks in the chunks")
    preserve_tables: bool = Field(..., description="Whether to preserve tables in the chunks")
    merge_short_sections: bool = Field(..., description="Whether to merge very short sections into adjacent chunks")
    metadata_focus: List[str] = Field(..., description="Which metadata fields to prioritize, e.g. ['page_title', 'section_title']"),
    notes: str = Field("",description="Any specific instructions for chunking this page")

def heuristic_router(record: dict) -> ChunkPlan:
    """Route page to chunking strategy based on URL and title patterns.
    
    Uses fast regex-based classification without LLM inference.
    
    Args:
        record: Crawled page record with url, metadata, and content
        
    Returns:
        ChunkPlan with optimized settings for page type
    """
    url = record.get("url", "unknown")
    title = (record.get("metadata") or {}).get("title") or ""
    title = title.lower()

    # Check if this is documentation
    if "/docs" in url or "/api/" in url or "/guides" in url or "/tutorials" in url or "/reference" in url or "documentation" in title:
        logger.debug("Classified as docs: %s", url)
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
    # Check if this is pricing page
    elif "pricing" in url or "plans" in url or "billing" in url or "pricing" in title:
        logger.debug("Classified as pricing: %s", url)
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

    # Check if this is a product/feature page
    elif "/product" in url or "/features" in url or "buy" in url or "product" in title or "features" in title:
        logger.debug("Classified as product: %s", url)
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
    # Check if this is a blog/article page
    elif "/blog" in url or "article" in url or "published" in url or "blog" in title:
        logger.debug("Classified as blog: %s", url)
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
    
    # Check if this is an FAQ/help page
    elif "/faq" in url or "/help" in url or "faq" in title:
        logger.debug("Classified as faq: %s", url)
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
    # Check if this is a legal/terms page
    elif "/legal" in url or "/terms" in url or "/privacy" in url:
        logger.debug("Classified as legal: %s", url)
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
    # Default fallback for unclassified pages
    else:
        logger.debug("Could not classify page, using default: %s", url)
        return ChunkPlan(
            page_type="general",
            chunk_style="thematic",
            target_chunk_words=300,
            preserve_code_blocks=False,
            preserve_tables=False,
            merge_short_sections=True,
            metadata_focus=["title","summary"],
            notes=""
        )


# Configure LLM for structured output
llm_chunk_planner = llm.with_structured_output(ChunkPlan)

def llm_planner(record: dict) -> ChunkPlan:
    """Use LLM to analyze page content and create optimal chunking plan.
    
    Called when heuristic routing cannot confidently classify a page.
    Uses page content sample for intelligent chunking decisions.
    
    Args:
        record: Crawled page record with url, metadata, and content
        
    Returns:
        ChunkPlan optimized for this specific page
    """
    url = record.get("url", "unknown")
    raw_html = record.get("content", "")[:3000]
    crawled_metadata = record.get("metadata", {})
    ChunkPlan_config = record.get("crawl_config", {})
    
    logger.info("Using LLM planner for: %s", url)
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
    logger.debug("LLM plan: type=%s style=%s words=%d", response.page_type, response.chunk_style, response.target_chunk_words)
    return response

def create_chunk_plan(record: dict) -> ChunkPlan:
    """Main entry point for chunk planning.
    
    Attempts heuristic routing first, then falls back to LLM if needed.
    On error, returns a safe default chunk plan.
    
    Args:
        record: Crawled page record to create a plan for
        
    Returns:
        ChunkPlan ready for use in chunking process
    """
    url = record.get("url", "unknown")
    try:
        # Try fast heuristic routing first
        heuristic_router_plan = heuristic_router(record)
        if heuristic_router_plan.page_type != "general":
            logger.info("Chunk plan created (heuristic): %s", url)
            return heuristic_router_plan
        else:
            # Fall back to LLM for complex cases
            logger.info("Heuristic inconclusive, using LLM planner: %s", url)
            return llm_planner(record)
    except Exception as e:
        logger.error("Chunk planning failed for %s", url, exc_info=True)
        logger.warning("Using default fallback chunk plan")
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