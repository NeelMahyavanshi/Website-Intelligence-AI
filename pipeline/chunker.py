from pydantic import BaseModel, Field
from pipeline.prompts.chunk_prompt import chunk_prompt as SYSTEM_PROMPT
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline.prompts.build_chunk_prompt import base_prompt, templates 
from typing import Dict, Any
from pipeline.chunk_planner import create_chunk_plan
import os
load_dotenv(override=True)
import re
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


class ChunkMetadata(BaseModel):
    page_title: str
    section_title: str
    summary: str
    keywords: list[str]
    entities: list[str]
    content_type: str = Field(..., description="Type of content: guide, product, docs, blog, other", examples=["guide"])
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="additional metadata fields specific to this chunk")

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class ChunkOutput(BaseModel):
    chunks: list[Chunk]

# llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"), temperature=0)

print(f"[CHUNKER] Initializing LLM with model: {os.getenv('OPENROUTER_MODEL')}")

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    temperature=0,
)
def build_chunk_prompt(plan):

    type_prompt = templates.get(plan.page_type, templates["general"])

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

    return f"{base_prompt}\n{type_prompt}\nAdditional Rules:\n{modifier_text}"



def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def validate_chunks(chunks: list[dict]) -> list[dict]:

    validated = []
    seen_hashes = set()

    for chunk in chunks:

        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        # 1. empty
        if not text:
            continue

        # 2. too short
        if len(text) < 80:
            continue

        # 3. too long
        if len(text) > 4000:
            continue

        # 4. duplicate
        h = text_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # 5. metadata defaults
        metadata.setdefault("page_title", "unknown")
        metadata.setdefault("section_title", "")
        metadata.setdefault("summary", text[:160])
        metadata.setdefault("keywords", [])
        metadata.setdefault("entities", [])
        metadata.setdefault("content_type", "general")
        metadata.setdefault("extra_metadata", {})

        # 6. keyword cleanup
        metadata["keywords"] = list({
            k.strip().lower()
            for k in metadata.get("keywords", [])
            if k and isinstance(k, str)
        })

        chunk["metadata"] = metadata
        validated.append(chunk)

    return validated


def fallback_chunk_page(content: str) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    texts = splitter.split_text(content)

    chunks = []

    for i, text in enumerate(texts, start=1):
        chunks.append({
            "text": text,
            "metadata": {
                "page_title": "unknown",
                "section_title": f"fallback_chunk_{i}",
                "summary": text[:180],
                "keywords": [],
                "entities": [],
                "content_type": "general",
                "extra_metadata": {
                    "generated_by": "recursive_splitter"
                }
            }
        })

    return chunks

def call_llm_chunker(content: str, prompt: str) -> ChunkOutput:

    model_with_structure = llm.with_structured_output(ChunkOutput)

    response = model_with_structure.invoke([
        ("system", prompt),
        ("human", content)
    ])
    return response

def process_record(record: dict) -> list[dict]:

    plan = create_chunk_plan(record)

    prompt = build_chunk_prompt(plan)

    url = record.get("url", "unknown")
    raw_html = record.get("content", "")[:12000]
    crawled_metadata = record.get("metadata", {})
    timestamp = record.get("timestamp", "unknown")

    llm_response = call_llm_chunker(raw_html, prompt)
    chunks = llm_response.chunks

    chunks = [c.model_dump() for c in chunks]
    chunks = validate_chunks(chunks)
    
    enriched_chunks = []
    for i,chunk in enumerate(chunks):
        chunk_dict = chunk.model_dump()
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

    validated_chunks = validate_chunks(chunks)

    return enriched_chunks

