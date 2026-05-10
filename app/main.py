from fastapi import FastAPI
from app.routes import health, ingest, query, companies
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Website Intelligence API",
    description="API for ingesting website data and querying it using LLMs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(companies.router, prefix="", tags=["Companies"])