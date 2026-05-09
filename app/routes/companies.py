from fastapi import APIRouter
from utils.database import db
from utils.logger import get_logger
from pipeline.store import _safe_id, _collection_cache
import chromadb
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logger = get_logger("COMPANIES_ROUTE")

router = APIRouter()

@router.get("/companies")
async def get_companies() -> list:

    response = db.table("ingest_jobs").select("company_id, start_url, status, company_type").execute()

    logger.info("Fetched ingest jobs for companies: %s", response.data)

    distinct_companies = []

    if response.data: 

        seen = set()
        for row in response.data:
            if row["company_id"] not in seen:
                seen.add(row["company_id"])
                distinct_companies.append(row)
            
            else:
                continue
    else:
        return []
    
    logger.info("Fetched distinct companies: %s", distinct_companies)
    return distinct_companies


@router.delete("/companies/{company_id}")
async def companies_delete(company_id: str):
    """Deletes all data associated with a company. This includes:
    1. All chunks in the "chunks" table with the given company_id
    2. All pages in the "crawled_pages" table with the given company_id
    3. All ingest jobs in the "ingest_jobs" table with the given company_id
    4. The corresponding collection in the vector database
    """
    
    logger.info("Deleting data for company: %s", company_id)

    try:

        # Delete records from DB tables
        db.table("chunks").delete().eq("company_id", company_id).execute()
        db.table("crawled_pages").delete().eq("company_id", company_id).execute()
        db.table("ingest_jobs").delete().eq("company_id", company_id).execute()
        logger.info("Deleted company data from DB for company: %s", company_id)

        # Delete collection from vector DB
        full_collection_name = f"{_safe_id(company_id)}_web_docs"
        client = chromadb.CloudClient(
            tenant=os.getenv("CHROMADB_TENANT"),
            database="main",
            api_key=os.getenv("CHROMADB_API_KEY"),
        )
        client.delete_collection(full_collection_name)
        logger.info("Deleted collection from vector DB: %s", full_collection_name)

        # Also remove from local cache if exists
        _collection_cache.pop(f"{company_id}_web_docs", None)
        logger.info("Deleted company data and cleared cache for company: %s", company_id)
        return {"status": "success", "message": f"Deleted all data for company {company_id}"}

    except Exception as e:
        logger.error("Error occurred while deleting company data from DB for company: %s", company_id)
        logger.error("Error details: %s", str(e), exc_info=True)
        return {"status": "error", "message": str(e)}
    
    

    


    


