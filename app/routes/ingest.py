from fastapi import APIRouter
from pydantic import BaseModel
from pipeline.store import ingest

router = APIRouter()

class IngestRequest(BaseModel):
    url: str
    company_id: str
    company_type: str

@router.post("/")
async def ingest_endpoint(request: IngestRequest):
    """
    Endpoint to handle website ingestion. It takes a URL and company information, processes the website content, 
    and stores it in the vector database for later retrieval.
    """
    try:
        count = await ingest(request.url, request.company_id, request.company_type)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}