from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pipeline.ingest_pipeline import run_ingest
from utils.database import db
from pipeline.ingest_pipeline import create_ingest_job

router = APIRouter()

class IngestRequest(BaseModel):
    url: str

@router.post("/")
async def ingest_endpoint(request: IngestRequest, background_tasks : BackgroundTasks):
    """
    Endpoint to handle website ingestion. It takes a URL and company information, processes the website content, 
    and stores it in the vector database for later retrieval.
    """
    try:

        result = create_ingest_job(request.url)
        job_id = result["job_id"]
        background_tasks.add_task(run_ingest, request.url, job_id)
        return {"status": "started", "message": "Ingestion started in background", "job_id" : job_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}   
    
@router.get("/{job_id}")
async def ingest_status(job_id:str):
    job = db.table("ingest_jobs").select("*").eq("id", job_id).execute()
    if not job.data:
        return {"status": "not_found"}
    return job.data[0]