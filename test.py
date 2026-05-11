import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

from pipeline.ingest_pipeline import run_chunking, run_embedding

async def main():
    url = "https://docs.crawl4ai.com"
    
    print("Resuming chunking...")
    chunk_result = await run_chunking(url)
    print(chunk_result)
    
    print("Resuming embedding...")
    embed_result = await run_embedding(url)
    print(embed_result)

asyncio.run(main())