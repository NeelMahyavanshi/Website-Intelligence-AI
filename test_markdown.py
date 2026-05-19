"""
Test file to compare markdown outputs from Crawl4AI:
- result.markdown (raw object)
- result.markdown.raw_markdown (unfiltered)
- result.markdown.fit_markdown (pruned/filtered)
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

TEST_URL = "https://docs.pipecat.ai/pipecat/learn/session-initialization"

async def test_markdown():
    prune_filter = PruningContentFilter(
        threshold=0.4,
        threshold_type="fixed",
        min_word_threshold=10
    )
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(TEST_URL, config=config)

        if not result.success:
            print(f"Crawl failed: {result.error_message}")
            return

        # ── raw_markdown ──────────────────────────────────────
        print("\n" + "="*60)
        print("RAW MARKDOWN (unfiltered)")
        print("="*60)
        print(f"Length: {len(result.markdown.raw_markdown)} chars")
        print("\n--- SAMPLE (first 2000 chars) ---\n")
        print(result.markdown.raw_markdown[:2000])

        # ── fit_markdown ──────────────────────────────────────
        print("\n" + "="*60)
        print("FIT MARKDOWN (after PruningContentFilter)")
        print("="*60)
        print(f"Length: {len(result.markdown.fit_markdown)} chars")
        print("\n--- SAMPLE (first 2000 chars) ---\n")
        print(result.markdown.fit_markdown[:2000])

        # ── result.markdown (direct) ──────────────────────────
        print("\n" + "="*60)
        print("RESULT.MARKDOWN (direct object)")
        print("="*60)
        print(f"Type: {type(result.markdown)}")
        markdown_str = str(result.markdown)
        print(f"Length: {len(markdown_str)} chars")
        print("\n--- SAMPLE (first 2000 chars) ---\n")
        print(markdown_str[:2000])

        # ── side-by-side code block check ─────────────────────
        print("\n" + "="*60)
        print("CODE BLOCK CHECK")
        print("="*60)
        raw_code_blocks = result.markdown.raw_markdown.count("```")
        fit_code_blocks = result.markdown.fit_markdown.count("```")
        result_code_blocks = str(result.markdown).count("```")
        print(f"Code fences in raw_markdown:     {raw_code_blocks}")
        print(f"Code fences in fit_markdown:     {fit_code_blocks}")
        print(f"Code fences in result.markdown:  {result_code_blocks}")
        print(f"Code blocks lost by pruning:     {(raw_code_blocks - fit_code_blocks) // 2}")

        # ── save to files for full comparison ─────────────────
        with open("test_raw_markdown.md", "w") as f:
            f.write(result.markdown.raw_markdown)

        with open("test_fit_markdown.md", "w") as f:
            f.write(result.markdown.fit_markdown)

        with open("test_result_markdown.md", "w") as f:
            f.write(str(result.markdown))

        print("\nFull outputs saved to:")
        print("  test_raw_markdown.md")
        print("  test_fit_markdown.md")
        print("  test_result_markdown.md")


if __name__ == "__main__":
    asyncio.run(test_markdown())
