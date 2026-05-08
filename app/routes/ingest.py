from fastapi import APIRouter
from pydantic import BaseModel
from pipeline.ingest_pipeline import run_ingest

router = APIRouter()

class IngestRequest(BaseModel):
    url: str

@router.post("/")
async def ingest_endpoint(request: IngestRequest):
    """
    Endpoint to handle website ingestion. It takes a URL and company information, processes the website content, 
    and stores it in the vector database for later retrieval.
    """
    try:
        result = await run_ingest(request.url)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}   