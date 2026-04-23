"""Website Intelligence AI - Chunker Module

Handles chunking of crawled web pages into semantically meaningful pieces.
Uses LLM-based intelligent splitting with validation and enrichment.
"""

from pydantic import BaseModel, Field
from pipeline.prompts.chunk_prompt import chunk_prompt as SYSTEM_PROMPT
from dotenv import load_dotenv
from pipeline.prompts.build_chunk_prompt import base_prompt, templates 
from typing import Dict, Any
from pipeline.chunk_planner import create_chunk_plan
from llm_model import llm
import os
load_dotenv(override=True)
import re
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("[CHUNKER] Module loaded successfully")


class ChunkMetadata(BaseModel):
    """Metadata associated with each chunk.
    
    Contains important fields for retrieval, ranking, and context.
    """
    page_title: str
    section_title: str
    summary: str
    keywords: list[str]
    entities: list[str]
    content_type: str = Field(..., description="Type of content: guide, product, docs, blog, other", examples=["guide"])
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="additional metadata fields specific to this chunk")

class Chunk(BaseModel):
    """Individual chunk of content.
    
    Contains the text and associated metadata.
    """
    text: str
    metadata: ChunkMetadata

class ChunkOutput(BaseModel):
    """Output schema for LLM chunking response."""
    chunks: list[Chunk]

print(f"[CHUNKER] Using shared LLM instance from llm_model")

def build_chunk_prompt(plan):
    """Build system prompt for chunking based on chunk plan.
    
    Dynamically creates instructions based on page type and settings.
    
    Args:
        plan: ChunkPlan object with chunking strategy
        
    Returns:
        String containing the complete system prompt for LLM
    """
    print(f"[CHUNKER] Building chunk prompt for page_type={plan.page_type}...")
    
    # Get template based on page type
    type_prompt = templates.get(plan.page_type, templates["general"])

    # Build modifier list based on plan settings
    modifiers = []

    if plan.preserve_code_blocks:
        modifiers.append("- Preserve all code fences exactly")

    if plan.preserve_tables:
        modifiers.append("- Preserve tables in readable markdown form")

    if plan.merge_short_sections:
        modifiers.append("- Merge very short adjacent sections")

    if plan.metadata_focus:
        fields = ", ".join(plan.metadata_focus)
        modifiers.append(f"- Prioritize metadata extraction for: {fields}")

    if plan.notes:
        modifiers.append(f"- Special notes: {plan.notes}")

    modifier_text = "\n".join(modifiers)
    print(f"[CHUNKER] ✓ Chunk prompt built with {len(modifiers)} modifiers")
    return f"{base_prompt}\n{type_prompt}\nAdditional Rules:\n{modifier_text}"



def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing, stripping, and collapsing whitespace.
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text string
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text: str) -> str:
    """Generate MD5 hash of normalized text for duplicate detection.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def validate_chunks(chunks: list[dict]) -> list[dict]:
    """Validate and clean chunks, removing invalid/duplicate entries.
    
    Performs multiple validation checks:
    - Empty chunks
    - Chunks too short (< 80 chars)
    - Chunks too long (> 4000 chars)
    - Duplicate detection via MD5 hashing
    - Metadata defaults and cleanup
    
    Args:
        chunks: List of chunk dictionaries to validate
        
    Returns:
        List of validated, cleaned chunks
    """
    print(f"[CHUNKER] Validating {len(chunks)} chunks...")
    validated = []
    seen_hashes = set()
    removed_count = 0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        # 1. Skip empty chunks
        if not text:
            removed_count += 1
            continue

        # 2. Skip chunks too short
        if len(text) < 25 and len(text.split()) < 5:
            removed_count += 1
            continue

        # 3. Skip chunks too long
        if len(text) > 4000:
            removed_count += 1
            continue

        # 4. Skip duplicate chunks based on text hash
        h = text_hash(text)
        if h in seen_hashes:
            removed_count += 1
            continue
        seen_hashes.add(h)

        # 5. Set metadata defaults
        metadata.setdefault("page_title", "unknown")
        metadata.setdefault("section_title", "")
        metadata.setdefault("summary", text[:160])
        metadata.setdefault("keywords", [])
        metadata.setdefault("entities", [])
        metadata.setdefault("content_type", "general")
        metadata.setdefault("extra_metadata", {})

        # 6. Clean up keywords - deduplicate and normalize
        metadata["keywords"] = list({
            k.strip().lower()
            for k in metadata.get("keywords", [])
            if k and isinstance(k, str)
        })

        chunk["metadata"] = metadata
        validated.append(chunk)

    print(f"[CHUNKER] ✓ Validation complete: {len(validated)} valid chunks, {removed_count} removed")
    return validated


def fallback_chunk_page(content: str) -> list[dict]:
    """Fallback chunking using simple recursive character splitting.
    
    Used when LLM chunking fails. Creates chunks using sensible separators.
    
    Args:
        content: Page content to chunk
        
    Returns:
        List of chunk dictionaries from recursive splitting
    """
    print(f"[CHUNKER] Performing fallback chunking using recursive splitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"[CHUNKER] Splitting content into chunks...")
    texts = splitter.split_text(content)

    chunks = []
    print(f"[CHUNKER] Creating {len(texts)} chunk objects...")

    for i, text in enumerate(texts, start=1):
        chunks.append({
            "text": text,
            "metadata": {
                "page_title": "unknown",
                "section_title": f"fallback_chunk_{i}",
                "summary": text[:160].strip(),
                "keywords": [],
                "entities": [],
                "content_type": "general",
                "extra_metadata": {
                    "generated_by": "recursive_splitter"
                }
            }
        })

    print(f"[CHUNKER] ✓ Fallback chunking complete: {len(chunks)} chunks created")
    return chunks

def call_llm_chunker(content: str, prompt: str) -> ChunkOutput:
    """Call LLM to chunk content based on given prompt.
    
    Args:
        content: Page content to chunk
        prompt: System prompt with chunking instructions
        
    Returns:
        ChunkOutput with list of chunks from LLM
    """
    print(f"[CHUNKER] Calling LLM chunker...")
    model_with_structure = llm.with_structured_output(ChunkOutput)

    response = model_with_structure.invoke([
        ("system", prompt),
        ("human", content)
    ])
    print(f"[CHUNKER] ✓ LLM returned {len(response.chunks)} chunks")
    return response

def process_record(record: dict) -> list[dict]:
    """Process a crawled page record through the chunking pipeline.
    
    Steps:
    1. Create chunk plan based on page type
    2. Build chunk prompt from plan
    3. Call LLM to chunk content
    4. Validate and clean chunks
    5. Enrich chunks with metadata
    
    Args:
        record: Crawled page record with url, content, and metadata
        
    Returns:
        List of enriched, validated chunk dictionaries
    """
    print(f"[CHUNKER] === PROCESSING RECORD ===")
    print(f"[CHUNKER] URL: {record.get('url', 'unknown')}")
    
    # Step 1: Get chunking strategy
    print(f"[CHUNKER] Creating chunk plan...")
    plan = create_chunk_plan(record)

    # Step 2: Build prompt from plan
    prompt = build_chunk_prompt(plan)

    url = record.get("url", "unknown")
    raw_html = record.get("content", "").strip()[:10000]
    crawled_metadata = record.get("metadata", {})
    timestamp = record.get("timestamp", "unknown")

    # Step 3: Call LLM chunker
    print(f"[CHUNKER] Calling LLM chunker with content sample...")
    llm_response = call_llm_chunker(raw_html, prompt)
    chunks = llm_response.chunks

    # Step 4: Convert and validate
    print(f"[CHUNKER] Converting and validating chunks...")
    chunks = [c.model_dump() for c in chunks]
    chunks = validate_chunks(chunks)
    
    # Step 5: Enrich with metadata
    print(f"[CHUNKER] Enriching chunks with metadata...")
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_dict = chunk.copy()

        chunk_dict["metadata"].update({
            "source_url": url,
            "timestamp": timestamp,
            "chunk_id": i + 1,
            "total_chunks": len(chunks),
            "depth": crawled_metadata.get("depth"),
            "parent_url": crawled_metadata.get("parent_url"),
            "og_title": crawled_metadata.get("title"),
            "og_description": crawled_metadata.get("description"),
        })

        enriched_chunks.append(chunk_dict)

    print(f"[CHUNKER] Returned {len(enriched_chunks)} chunks for {url}")
    print(f"[CHUNKER] === PROCESSING COMPLETE ===")

    return enriched_chunks

