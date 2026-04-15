import asyncio
from datetime import datetime
import json
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, PruningContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator



async def crawl_url(start_url, max_depth=2, max_pages=100) -> list[dict]:

    prune_filter = PruningContentFilter(
        threshold=0.45,           
        threshold_type="dynamic",  
        min_word_threshold=50      
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth, 
            include_external=False,
            max_pages=max_pages
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        markdown_generator=md_generator,
        scan_full_page=True
    )
    try:

        pages = []
        async with AsyncWebCrawler() as crawler:

            print(f"Starting crawl.....")

            results = await crawler.arun(start_url, config=config)

            print(f"Crawled {len(results)} pages in total")

            count = 0

            for result in results:  

                if not result.success or not result.markdown:
                    continue
                    
                data = {
                    "url": result.url,
                    "content": result.markdown.fit_markdown,
                    "metadata": result.metadata,
                    "timestamp":  datetime.now().isoformat()  
                }
                
                pages.append(data)
                count += 1

                print(f"✓ Saved {count} pages")

        return pages

    except Exception as e:
        print(f"Error during crawling: {e}")
