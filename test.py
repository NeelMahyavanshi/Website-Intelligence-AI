import asyncio
from datetime import datetime
import json
import hashlib
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

seen_hashes = set()

def hash_content(text):
    return hashlib.md5(text.encode()).hexdigest()

async def main(start_url, max_depth=2, max_pages=5, output_file="crawled_data.jsonl"):
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth, 
            include_external=False,
            max_pages=max_pages
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )
    try:

        async with AsyncWebCrawler() as crawler:

            print(f"Starting crawl. Data will be saved to {output_file}")

            results = await crawler.arun(start_url, config=config)

            print(f"Crawled {len(results)} pages in total")

            count = 0

            with open(output_file, "a", encoding="utf-8") as f:

                for result in results:  
                    
                    data = {
                                "url": result.url,
                                "markdown": result.markdown,
                                "metadata": result.metadata,
                                "timestamp":  datetime.now().isoformat()  
                            }
                    
                    
                    f.write(json.dumps(data) + "\n")
                    count += 1

                print(f"✓ Saved {count} pages")

    except Exception as e:
        print(f"Error during crawling: {e}")

if __name__ == "__main__":
    asyncio.run(main("https://docs.agno.com/"))