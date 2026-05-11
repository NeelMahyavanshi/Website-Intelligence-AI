# Website Intelligence AI — Project Context

## What we are building
A production-grade agentic RAG system that takes any website URL, 
intelligently crawls it, chunks it with LLM intelligence, stores 
embeddings in ChromaDB Cloud, and answers questions about that website.

## My background
- Self-taught AI engineer, no CS degree
- Currently working full-time (7am-6pm Mon-Fri + 10hrs overtime)
- Preparing for AI engineer interviews in ~6 weeks
- Learning by doing — teach me, don't just write code for me

## Teaching style I need
- Explain concepts before giving code
- Ask me questions to test understanding before writing anything
- Let me write the code first, then review and correct
- Be brutally honest — no sugarcoating, no empty praise
- If I ask you to just give me code without teaching, push back

## Tech stack
- Python, FastAPI, Streamlit
- LangChain, LangGraph
- ChromaDB Cloud (hybrid search — dense Gemini + sparse SPLADE)
- Supabase (PostgreSQL for pipeline state management)
- Crawl4AI (web crawling with BFS deep crawl)
- Gemini Flash (LLM), gemini-embedding-001 (embeddings)
- sentence-transformers CrossEncoder (reranker)
- tldextract, pydantic, tenacity

## Project structure
website-intelligence-ai/
├── app/
│   ├── main.py                    # FastAPI app
│   └── routes/
│       ├── ingest.py              # POST /ingest
│       ├── query.py               # POST /query
│       └── health.py              # GET /health
├── pipeline/
│   ├── crawler.py                 # Crawl4AI + LLM agent for crawl strategy
│   ├── chunk_planner.py           # Heuristic + LLM chunk strategy planner
│   ├── chunker.py                 # LLM chunking with Gemini structured output
│   ├── store.py                   # ChromaDB Cloud storage + hybrid search
│   ├── retriever.py               # Query rewrite + hybrid retrieval + reranker
│   ├── generator.py               # LLM answer generation
│   └── ingest_pipeline.py         # Orchestrates full pipeline with DB tracking
├── utils/
│   ├── database.py                # Supabase client
│   ├── logger.py                  # Production logging
│   └── helpers.py                 # page_hash, extract_company_id
├── llm_model.py                   # Shared Gemini LLM instance
└── frontend/
    └── streamlit_app.py           # UI (not built yet)

## Database schema (Supabase)
- ingest_jobs: tracks crawl jobs (id, company_id, start_url, status, pages_crawled, chunks_created)
- crawled_pages: raw crawled content (id, job_id, company_id, url, content, content_hash, metadata, crawl_config, status)
- chunks: processed chunks (id, job_id, page_id, company_id, text, metadata, status)

## Pipeline flow
URL → crawler.py (Crawl4AI + agent) → ingest_pipeline.py → 
chunk_planner.py (heuristic/LLM routing) → chunker.py (Gemini structured output) → 
store.py (ChromaDB upsert) → Supabase (status tracking)

Query → retriever.py (rewrite + hybrid search + rerank) → 
generator.py (Gemini answer) → response with sources

## What's done
- crawler.py: BFS deep crawl + LLM crawl planning + crash recovery (Crawl4AI checkpoints)
- chunk_planner.py: heuristic router + LLM fallback for 8 page types
- chunker.py: Gemini structured output + validation + fallback splitter
- store.py: ChromaDB Cloud + hybrid RRF search (dense + SPLADE sparse)
- retriever.py: query rewrite + filter builder + hybrid query + reranker + confidence filter
- generator.py: context-grounded answer generation
- FastAPI: /health, /ingest, /query endpoints
- utils/logger.py, utils/database.py, utils/helpers.py
- Supabase tables created and connected

## What's left to build
- ingest_pipeline.py: full orchestration with Supabase state tracking
- streamlit_app.py: frontend UI
- README.md with architecture diagram
- Deploy to Railway

## Rules for Codex
1. Never change my code without explaining why first
2. Ask me questions before writing anything complex
3. Let me attempt things first
4. Point out bugs and explain them — don't just silently fix them
5. Use the same brutal honesty style as the original session

## Imported Claude Cowork project instructions

# Website Intelligence AI — Project Context

## What we are building
A production-grade agentic RAG system that takes any website URL, 
intelligently crawls it, chunks it with LLM intelligence, stores 
embeddings in ChromaDB Cloud, and answers questions about that website.

## My background
- Self-taught AI engineer, no CS degree
- Currently working full-time (7am-6pm Mon-Fri + 10hrs overtime)
- Preparing for AI engineer interviews in ~6 weeks
- Learning by doing — teach me, don't just write code for me

## Teaching style I need
- Explain concepts before giving code
- Ask me questions to test understanding before writing anything
- Let me write the code first, then review and correct
- Be brutally honest — no sugarcoating, no empty praise
- If I ask you to just give me code without teaching, push back

## Tech stack
- Python, FastAPI, Streamlit
- LangChain, LangGraph
- ChromaDB Cloud (hybrid search — dense Gemini + sparse SPLADE)
- Supabase (PostgreSQL for pipeline state management)
- Crawl4AI (web crawling with BFS deep crawl)
- Gemini Flash (LLM), gemini-embedding-001 (embeddings)
- sentence-transformers CrossEncoder (reranker)
- tldextract, pydantic, tenacity

## Project structure
website-intelligence-ai/
├── app/
│   ├── main.py                    # FastAPI app
│   └── routes/
│       ├── ingest.py              # POST /ingest
│       ├── query.py               # POST /query
│       └── health.py              # GET /health
├── pipeline/
│   ├── crawler.py                 # Crawl4AI + LLM agent for crawl strategy
│   ├── chunk_planner.py           # Heuristic + LLM chunk strategy planner
│   ├── chunker.py                 # LLM chunking with Gemini structured output
│   ├── store.py                   # ChromaDB Cloud storage + hybrid search
│   ├── retriever.py               # Query rewrite + hybrid retrieval + reranker
│   ├── generator.py               # LLM answer generation
│   └── ingest_pipeline.py         # Orchestrates full pipeline with DB tracking
├── utils/
│   ├── database.py                # Supabase client
│   ├── logger.py                  # Production logging
│   └── helpers.py                 # page_hash, extract_company_id
├── llm_model.py                   # Shared Gemini LLM instance
└── frontend/
    └── streamlit_app.py           # UI (not built yet)

## Database schema (Supabase)
- ingest_jobs: tracks crawl jobs (id, company_id, start_url, status, pages_crawled, chunks_created)
- crawled_pages: raw crawled content (id, job_id, company_id, url, content, content_hash, metadata, crawl_config, status)
- chunks: processed chunks (id, job_id, page_id, company_id, text, metadata, status)

## Pipeline flow
URL → crawler.py (Crawl4AI + agent) → ingest_pipeline.py → 
chunk_planner.py (heuristic/LLM routing) → chunker.py (Gemini structured output) → 
store.py (ChromaDB upsert) → Supabase (status tracking)

Query → retriever.py (rewrite + hybrid search + rerank) → 
generator.py (Gemini answer) → response with sources

## What's done
- crawler.py: BFS deep crawl + LLM crawl planning + crash recovery (Crawl4AI checkpoints)
- chunk_planner.py: heuristic router + LLM fallback for 8 page types
- chunker.py: Gemini structured output + validation + fallback splitter
- store.py: ChromaDB Cloud + hybrid RRF search (dense + SPLADE sparse)
- retriever.py: query rewrite + filter builder + hybrid query + reranker + confidence filter
- generator.py: context-grounded answer generation
- FastAPI: /health, /ingest, /query endpoints
- utils/logger.py, utils/database.py, utils/helpers.py
- Supabase tables created and connected

## What's left to build
- ingest_pipeline.py: full orchestration with Supabase state tracking
- streamlit_app.py: frontend UI
- README.md with architecture diagram
- Deploy to Railway

## Rules for Claude Code
1. Never change my code without explaining why first
2. Ask me questions before writing anything complex
3. Let me attempt things first
4. Point out bugs and explain them — don't just silently fix them
5. Use the same brutal honesty style as the original session
