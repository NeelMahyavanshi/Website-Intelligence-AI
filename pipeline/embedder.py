from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
from dotenv import load_dotenv
load_dotenv(override=True)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768,
    task_type="retrieval_document"
    )

def embed_chunks(chunks:list[dict], batch_size:int=10) -> list[dict]:

    """
    Embeds chunks in batches to avoid rate limits.
    Returns chunks with 'embedding' field attached.
    """
    if not chunks:
        return []
    
    all_embedded_chunks = []
    for i in range(0,len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]

        try:

            vectors = embeddings.embed_documents(texts)

            for chunk, vector in zip(batch, vectors): 
                chunk["embedding"] = vector
                all_embedded_chunks.append(chunk)

        except Exception as e:
            print(f"Error occurred for batch {i // batch_size + 1} : {e}")
            continue

        if i + batch_size < len(chunks):
            time.sleep(1)  # Sleep for 1 second between batches to avoid hitting rate limits

    return all_embedded_chunks

if __name__ == "__main__":
    sample_chunks = [
    {
        "text": "AgentOS turns your agents into a production service you can deploy anywhere. It provides 50+ ready to use endpoints with SSE-compatible streaming. Sessions, memory, knowledge, and traces are stored in your database, ensuring no state bleed between users, agents, or sessions. Security is handled via JWT-based RBAC with hierarchical scopes.",
        "metadata": {
            "page_title": "What is AgentOS? - Agno",
            "section_title": "AgentOS Overview",
            "summary": "Introduction to AgentOS as a production runtime for agents.",
            "keywords": ["AgentOS", "SSE streaming", "JWT RBAC", "FastAPI"],
            "entities": ["AgentOS", "Agno SDK"],
            "content_type": "docs",
            "source_url": "https://docs.agno.com/agent-os/introduction",
            "timestamp": "2026-04-12T09:14:46.562464",
            "chunk_id": 1,
            "total_chunks": 2,
            "depth": 1,
            "parent_url": "https://docs.agno.com/",
            "og_title": "What is AgentOS? - Agno",
            "og_description": "The production runtime and control plane for multi-agent systems."
        }
    },
    {
        "text": "The AgentOS runtime exposes a set of APIs that power both the control plane and your AI products. The AgentOS runs as a container in your cloud, and the UI connects directly from the browser without proxies or data relays, ensuring your data stays completely private.",
        "metadata": {
            "page_title": "What is AgentOS? - Agno",
            "section_title": "Privacy and Infrastructure",
            "summary": "Explanation of AgentOS deployment model emphasizing data privacy.",
            "keywords": ["data privacy", "self-hosting", "cloud container"],
            "entities": ["AgentOS", "Agno"],
            "content_type": "docs",
            "source_url": "https://docs.agno.com/agent-os/introduction",
            "timestamp": "2026-04-12T09:14:46.562464",
            "chunk_id": 2,
            "total_chunks": 2,
            "depth": 1,
            "parent_url": "https://docs.agno.com/",
            "og_title": "What is AgentOS? - Agno",
            "og_description": "The production runtime and control plane for multi-agent systems."
        }
    }
]
    embedded_chunks = embed_chunks(sample_chunks)

    for chunk in embedded_chunks:
        print(f"Text: {chunk['text'][:80]}...")
        print(f"Embedding dims: {len(chunk['embedding'])}")
        print(f"First 5 values: {chunk['embedding'][:5]}\n")
