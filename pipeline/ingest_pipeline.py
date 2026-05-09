"""
Ingest Pipeline

"""

from pipeline.crawler import crawl_url
from pipeline.chunker import process_record
from utils.helpers import extract_company_id
from utils.database import db
from utils.helpers import page_hash
from utils.logger import get_logger
from pipeline.embedder import run_embedding

logger = get_logger("INGEST_PIPELINE")


# ============================================================
# CREATE INJEST JOB_ID
# ============================================================

def create_ingest_job(start_url:str) -> str:

    """Creates a new ingest job record in the database and returns the job ID."""

    logger.info("Creating ingest job for URL: %s", start_url)

    response = db.table("ingest_jobs")\
        .insert({
            "company_id": extract_company_id(start_url),
            "start_url": start_url,
            "status": "in_progress"
        })\
        .execute()
    
    job_id = response.data[0]["id"]

    logger.info("Ingest job created with ID: %s", job_id)

    return {
        "job_id" : job_id
    }


# ============================================================
# SAVE_PAGES_TO_DB
# ============================================================

def save_pages_to_db(page: dict, job_id: str) -> str:
    """Saves crawled page data to the database.
    
    Args:
        page: A dictionary containing url, content, metadata, and timestamp
        job_id: The ID of the crawling job this page belongs to
    """
    page_record = {
        "job_id": job_id,
        "company_id": extract_company_id(page.get("url", "unknown")),
        "url": page.get("url", ""),
        "content": page.get("content", ""),
        "content_hash": page_hash(page.get("content", "")),
        "metadata": page.get("metadata", {}),
        "crawl_config": page.get("crawl_config", {}),
        "status": "chunking_pending",
    }

    try:

        content_hash_list = (
            db.table("crawled_pages")
            .select("content_hash")
            .eq("content_hash", page_record["content_hash"])
            .execute()
        )

        if content_hash_list and content_hash_list.data:
            logger.info("Duplicate page detected, skipping DB insert: %s", page_record["url"])
            return "duplicate"
        
        response = db.table("crawled_pages").insert(page_record).execute()
        logger.info("Page saved to DB: %s", page_record["url"])
        page_id = response.data[0]["id"]
        return page_id

    except Exception as e:
        logger.error("Failed to save page to DB: %s", page.get("url", "unknown"), exc_info=True)
        return "error"
    

# ============================================================
#  RUN CRAWL
# ============================================================

async def run_crawl(start_url: str, job_id: str) -> str:
    """Main function to run the ingest pipeline.
    
    Steps:
    1. Create a new ingest job record in the database
    2. Start crawling from the given URL
    3. For each crawled page, save to DB and process for chunking
    4. Update job status and return job_id
    """

    # Step 1: Create job record
    # Create job record in ingest_jobs → get job_id

    pages_crawled = 0
    chunks_created = 0

    logger.info("Starting ingest pipeline for URL: %s", start_url)

    # Step 1: Start crawling

    company_type = "default"

    async for page in crawl_url(start_url):

        # For each page: hash check → save to crawled_pages

        save_result = save_pages_to_db(page, job_id)

        company_type = page["crawl_config"].get("site_type", "default")

        if save_result == "error":
            logger.error("Error saving page to DB, skipping chunking: %s", page.get("url", "unknown"))
            continue
        elif save_result == "duplicate":
            logger.info("Page already exists in DB, skipping chunking: %s", page.get("url", "unknown"))
            continue
            
        pages_crawled += 1

    db.table("ingest_jobs")\
        .update({
            "status": "completed",
            "pages_crawled": pages_crawled,
            "chunks_created": chunks_created,
            "company_type": company_type
        })\
        .eq("id", job_id)\
        .execute()
    
    return {
        "job_id": job_id,
        "company_id": extract_company_id(start_url),
        "company_type": company_type,
        "status": "completed",
        "pages_crawled": pages_crawled,
        "chunks_created": chunks_created
    }

# ============================================================
#  RUN CHUNKING
# ============================================================
async def run_chunking(url: str) -> dict:
    """Runs the chunking process for all pages of a completed crawl job."""
        
    company_id = extract_company_id(url)

    # Get all pages with status "chunking_pending" and "company_id" for this job_id
    pages_to_chunk = db.table("crawled_pages")\
        .select("*")\
        .eq("company_id", company_id)\
        .eq("status", "chunking_pending")\
        .execute()
    
    if not pages_to_chunk.data:
        logger.warning("No pages found for chunking for company ID: %s", company_id)
        return {"status": "error", "message": "No pages found for chunking for this company"}
    logger.info("Starting chunking for company ID: %s, pages to chunk: %d", company_id, len(pages_to_chunk.data))

    chunks_created = 0

    for page in pages_to_chunk.data:

        # Process page through chunking pipeline
        try:
            chunks = process_record(page)
            for chunk in chunks:
                chunk_record = {
                    "job_id": page.get("job_id"),
                    "page_id": page.get("id"),
                    "company_id": extract_company_id(page.get("url", "unknown")),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "status": "ready_for_embedding",
                    "company_type": page["crawl_config"].get("site_type", "default")
                }
                db.table("chunks").insert(chunk_record).execute()
            logger.info("Chunks saved to DB for page: %s, chunks_count=%d", page.get("url", "unknown"), len(chunks))
            chunks_created += len(chunks)
        except Exception as e:
            logger.error("Error processing page for chunking: %s", page.get("url", "unknown"), exc_info=True)
            continue

        db.table("crawled_pages")\
            .update({"status": "chunked"})\
            .eq("id", page.get("id"))\
            .execute()
    
    logger.info("Chunking completed for company ID: %s, total chunks created: %d", company_id, chunks_created)

    return {
    "company_id": company_id,
    "status": "chunked",
    "chunks_created": chunks_created
    }


# ============================================================
# RUN INGEST
# ============================================================

async def run_ingest(url: str, job_id: str) -> dict:
    """Main entry point for the ingest pipeline.
    
    Args:
        url: The starting URL to crawl and ingest
        job_id: The ID of the ingest job created for this URL
    Returns:
        A dictionary containing the ingest job ID and status    
    """
    try:

        results = await run_crawl(url, job_id)
        company_id = results.get("company_id", "Unknown")
        chunks = await run_chunking(url)
        embedding = await run_embedding(url)
        return {
            "company_id": company_id,
            "company_type": results.get("company_type", "default"),
            "chunks_embedded": embedding.get("chunks_embedded"),
            "pages_crawled": results.get("pages_crawled"),
            "chunks_created": chunks.get("chunks_created"),
            "status": "completed"
        }
    except Exception as e:
        logger.error("Ingest pipeline failed for URL: %s", url, exc_info=True)
        return {"status": "error", "message": str(e)}
    