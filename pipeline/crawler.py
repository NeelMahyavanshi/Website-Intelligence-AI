import asyncio
from datetime import datetime
import json
from crawl4ai import AsyncUrlSeeder, AsyncWebCrawler, CrawlerRunConfig, PruningContentFilter,SeedingConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# ─── Get total active pages ───────────────────────────────────────────

async def get_total_active_pages(domain_url: str) -> int:
    try:
        config = SeedingConfig(source="sitemap+cc", live_check=True, concurrency=20)
        async with AsyncUrlSeeder() as seeder:
            discovered_urls = await seeder.urls(domain_url, config)
            active_pages = [url for url in discovered_urls if url.get("status") == "valid"]
            return len(active_pages)
    except Exception as e:
        print(f"URL seeding failed: {e}, returning 0")
        return 0
    
# ─── Agent output schema ───────────────────────────────────────────


class CrawlPlan_config(BaseModel):
    site_type: str = Field(..., description="One of: docs, ecommerce, blog, support, unknown")
    max_depth: int = Field(..., description="How deep to crawl. 1=surface only, 3=deep docs site")
    max_pages: int = Field(..., description="Max pages to crawl. Small site=20, large docs=100")
    pruning_threshold: float = Field(..., description="Threshold for pruning content. 0.3 to 0.8")
    notes: str = Field(..., description="Any specific instructions for crawling this site")

llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"), temperature=0)

async def plan_crawl(url:str) -> CrawlPlan_config:
    """
    Agent inspects the URL and decides how to crawl it.
    This is the 'brain' — runs before any crawling starts.
    """
    
    structured_llm = llm.with_structured_output(CrawlPlan_config)

    number_of_pages = await get_total_active_pages(url)

    response = structured_llm.invoke(f"""
    Analyze this website URL and create best crawling strategy.

    URL: {url}

    As per the latest scan, this website has approximately {number_of_pages} active pages.

    Rules:
    - Documentation sites (docs., /docs, /guide, /reference): depth=3, pages=80
    - E-commerce sites: depth=2, pages=50, likely JS-heavy
    - Blogs or marketing sites: depth=2, pages=30
    - Support/help centers: depth=2, pages=60
    - Unknown or simple sites: depth=1, pages=20
    - Any site using React/Next.js/Vue likely needs JS rendering

    Analyze the URL pattern and decide:

    - site_type (docs, ecommerce, blog, support, app)
    - max_depth (1-5)
    - max_pages (20-500)
    - pruning_threshold (0.3 to 0.8)
    - notes (any specific instructions for crawling this site)

    Return structured output only.
    """)
    print(f"Agent decision: {response.site_type} site, max_depth={response.max_depth}, max_pages={response.max_pages}, pruning_threshold={response.pruning_threshold}")
    print(f"Agent notes: {response.notes}")
    return response

# ─── Main crawl function ───────────────────────────────────────────


async def crawl_url(start_url:str) -> list[dict]:

    """
    Crawls a URL using agent-decided configuration.
    Agent runs first, then Crawl4AI executes the strategy.
    """

    try:
        plan = await plan_crawl(start_url)
    except Exception as e:
        print(f"Agent planning failed: {e}, using fallback config")
        plan = CrawlPlan_config(
            site_type="default",
            max_depth=2,
            max_pages=100,
            pruning_threshold=0.45,
            notes="fallback"
        )

    print(f"Crawl plan: {plan}")

    prune_filter = PruningContentFilter(
        threshold=plan.pruning_threshold,           
        threshold_type="dynamic",  
        min_word_threshold=50      
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=plan.max_depth, 
            include_external=False,
            max_pages=plan.max_pages
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
                    "timestamp":  datetime.now().isoformat(),
                    "crawl_config" : {
                        "site_type": plan.site_type,
                        "max_depth": plan.max_depth,
                        "max_pages": plan.max_pages,
                        "pruning_threshold": plan.pruning_threshold,
                        "notes": plan.notes
                    }
                }
                
                pages.append(data)
                count += 1

                print(f"✓ Saved {count} pages")

        return pages

    except Exception as e:
        print(f"Error during crawling: {e}")
