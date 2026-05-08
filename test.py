import asyncio 
from pipeline.ingest_pipeline import run_ingest

async def test_run_ingest():
    test_url = "https://crawl4ai.com"
    result = await run_ingest(test_url)
    print(result)

if __name__ == "__main__":
    asyncio.run(test_run_ingest())