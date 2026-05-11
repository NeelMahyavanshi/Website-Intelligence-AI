from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from utils.database import db
from pipeline.ingest_pipeline import run_ingest, create_ingest_job, resume_pipeline
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
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{job_id}")
async def ingest_status(job_id:str):
    job = db.table("ingest_jobs").select("*").eq("id", job_id).execute()
    if not job.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.data[0]

@router.post("/resume/{job_id}")
async def resume_jobs(job_id:str,background_tasks : BackgroundTasks):
    job = db.table("ingest_jobs").select("*").eq("id", job_id).execute()
    if not job.data:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.data[0]["status"] == "completed":
        raise HTTPException(status_code=400, detail="nothing to resume)")
    else:
        background_tasks.add_task(resume_pipeline, job.data[0]["start_url"], job.data[0]["id"], job.data[0]["status"])
        return {"status": "resuming", "job_id": job_id}
        