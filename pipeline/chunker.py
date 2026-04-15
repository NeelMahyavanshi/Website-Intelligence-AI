from pydantic import BaseModel, Field
from chunk_prompt import SYSTEM_PROMPT
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
load_dotenv(override=True)


class ChunkMetadata(BaseModel):
    page_title: str
    section_title: str
    summary: str
    keywords: list[str]
    entities: list[str]
    content_type: str = Field(..., description="Type of content: guide, product, docs, blog, other", examples=["guide"])

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class ChunkOutput(BaseModel):
    chunks: list[Chunk]

llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"), temperature=0)


def call_llm_chunker(state: dict) -> ChunkOutput:

    model_with_structure = llm.with_structured_output(ChunkOutput)

    response = model_with_structure.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", state['raw_html'])
    ])
    return response

def process_record(record: dict) -> list[dict]:
    url = record.get("url", "unknown")
    raw_html = record.get("content", "")
    crawled_metadata = record.get("metadata", {})
    timestamp = record.get("timestamp", "unknown")

    llm_response = call_llm_chunker({"raw_html": raw_html})
    chunks = llm_response.chunks

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
    return enriched_chunks


# with open("crawled_data.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             record = json.loads(line)
#         except json.JSONDecodeError:
#             print("Invalid JSON line, skipping...")
#             continue
#         if record.get("content") == "" or len(record.get("content", "")) < 10:
#             print("No content, skipping...")
#             continue
#         print(f"Processing: {record.get('url')}")
#         enriched_chunks = process_record(record)
#         print(enriched_chunks)