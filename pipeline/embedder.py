from pipeline.store import safe_upsert
from utils.logger import get_logger
from utils.database import db
from pipeline.store import get_collection
from utils.helpers import extract_company_id
from langsmith import traceable

logger = get_logger("EMBEDDER")

# ============================================================
# RUN EMBEDDING
# ============================================================

async def run_embedding(url: str) -> dict:
    """Runs the embedding process for all chunks of a company.
    1. Get all chunks with status "ready_for_embedding" and "company_id"
    2. For each chunk, generate embedding and upsert to vector DB
    3. Update chunk status to "embedded" in DB
    """

    company_id = extract_company_id(url)

    ready_to_embbed = db.table("chunks")\
                        .select("*")\
                        .eq("status", "ready_for_embedding")\
                        .eq("company_id", company_id)\
                        .execute()
    
    if not ready_to_embbed.data:
        logger.warning("No pages found for embedding for company ID: %s", company_id)
        return {"status": "error", "message": "No pages found for embedding for this company"}
    
    logger.info("Starting embedding for company ID: %s, pages to chunk: %d", company_id, len(ready_to_embbed.data))

    embedded = 0

    company_type = ready_to_embbed.data[0].get("company_type", "default")
    collection = get_collection(company_id, "web_docs", company_type)

    for embed in ready_to_embbed.data:

        try:

            logger.info("Processing chunk ID: %s", embed.get("id"))

            safe_upsert(collection, [embed])
            db.table("chunks")\
                .update({"status": "embedded"})\
                .eq("id", embed.get("id"))\
                .execute()
            
            embedded += 1

            logger.info("Chunk embedded and status updated for chunk ID: %s", embed.get("id"))


        except Exception as e:
            logger.error("Error occurred while processing chunk ID: %s", embed.get("id"))
            logger.error("Error details: %s", str(e), exc_info=True)
            continue
    
    if embedded == 0:
        logger.warning("No chunks were embedded for company ID: %s", company_id)
        return {"status": "error", "message": "No chunks were embedded for this company"}

    logger.info("Embedding completed for company ID: %s, total chunks embedded: %d", company_id, embedded)

    return {
        "company_id": company_id,
        "company_type": company_type,
        "status": "embedded",
        "chunks_embedded": embedded
    }