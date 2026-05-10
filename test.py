import asyncio 
from pipeline.ingest_pipeline import run_crawl, create_ingest_job, run_chunking, run_embedding

async def test_run_crawl():
    test_url = "https://docs.crawl4ai.com/core/deep-crawling/#32-streaming-mode"
    job = create_ingest_job(test_url)
    job_id = job["job_id"]
    result = await run_crawl(test_url, job_id)
    print(result)

async def test_run_chunking():
    test_url = "https://docs.crawl4ai.com/core/deep-crawling/#32-streaming-mode"
    result = await run_chunking(test_url)
    print(result)

async def test_run_embedding():
    test_url = "https://docs.crawl4ai.com/core/deep-crawling/#32-streaming-mode"
    result = await run_embedding(test_url)
    print(result)

if __name__ == "__main__":
    # asyncio.run(test_run_crawl())
    asyncio.run(test_run_chunking())
    asyncio.run(test_run_embedding())

from pipeline.retriever import retrieve
from pipeline.generator import generate

def test_query():
    query = "What are the benefits of using streaming mode in Crawl4AI's deep crawling?"  
    company_id = "crawl4ai"
    retrieval = retrieve(query, company_id)
    print("=== CONTEXT ===")
    print(retrieval["context"])
    print("=== END CONTEXT ===")
    result = generate(query, retrieval)
    print(result)

