from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pipeline.retriever import retrieve
from pipeline.generator import generate
from utils.helpers import extract_company_id
import asyncio
from utils.logger import get_logger

logger = get_logger("QUERY_ROUTE")

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    url:str

@router.post("/")
async def query_endpoint(request: QueryRequest):

    """
    Handles incoming query requests. It takes a user query and company information, retrieves relevant data from the vector database,
    and generates a response using the LLM.
    """

    company_id = None
    try:
        loop = asyncio.get_event_loop()
        company_id = extract_company_id(request.url)
        retrieval = await loop.run_in_executor(
            None, retrieve, request.query, company_id
        )
        generation = await loop.run_in_executor(
            None, generate, request.query, retrieval
        )
        return generation
    except Exception as e:
        if isinstance(e, ValueError) and "No collection found" in str(e):
            logger.warning("No data found for company ID: %s", company_id)
            raise HTTPException(status_code=404, detail="No data found for this company")
        logger.error("Error processing query: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    


    