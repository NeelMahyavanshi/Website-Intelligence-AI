from fastapi import APIRouter
from pydantic import BaseModel
import asyncio

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    url:str
    company_type: str

@router.post("/")
async def query_endpoint(request: QueryRequest):

    """
    Handles incoming query requests. It takes a user query and company information, retrieves relevant data from the vector database,
    and generates a response using the LLM.
    """
    from pipeline.retriever import retrieve
    from pipeline.generator import generate
    from utils.helpers import extract_company_id
    try:
        loop = asyncio.get_event_loop()
        company_id = extract_company_id(request.url)
        retrieval = await loop.run_in_executor(
            None, retrieve, request.query, company_id, request.company_type
        )
        generation = await loop.run_in_executor(
            None, generate, request.query, retrieval
        )
        return generation
    except Exception as e:
        return {"status": "error", "message": str(e)}