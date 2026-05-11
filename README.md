# Website Intelligence AI

A production-grade **agentic RAG (Retrieval-Augmented Generation) system** that intelligently crawls any website, chunks content with LLM intelligence, stores embeddings in ChromaDB Cloud, and answers questions about that website with sources.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green) ![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5+-purple) ![License](https://img.shields.io/badge/License-MIT-blue)

## 🎯 What This Does

Website Intelligence AI is a complete pipeline for ingesting website data and making it queryable via an intelligent Q&A interface:

1. **Smart Crawling** — Uses Crawl4AI with LLM-powered strategy selection for efficient BFS deep crawling
2. **Intelligent Chunking** — Leverages Gemini LLM + heuristic routing to split content based on page type (docs, blogs, FAQs, pricing, etc.)
3. **Hybrid Search** — Combines dense (Gemini embeddings) + sparse (SPLADE) vectors with RRF ranking
4. **LLM-Powered Q&A** — Rewrites queries, retrieves context, reranks results, generates grounded answers
5. **Production Monitoring** — LangSmith tracing for every pipeline stage, Supabase for state tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INGEST PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  URL → [crawler.py] → [chunk_planner.py] → [chunker.py] →      │
│         (Crawl4AI      (8 page types)      (Gemini structured   │
│          + BFS)        (heuristic/LLM)      output)             │
│                                              ↓                   │
│                                    [store.py] (ChromaDB Cloud)   │
│                                              ↓                   │
│                                    [embedder.py] (Gemini)       │
│                                              ↓                   │
│                                    [ingest_pipeline.py] (DB)    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      QUERY PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query → [retriever.py]          → [generator.py] → Answer     │
│            ├─ Rewrite (LLM)          (Gemini)      + Sources   │
│            ├─ Hybrid Search (dense + sparse)                   │
│            ├─ Rerank (CrossEncoder)                             │
│            └─ Format context                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Database: Supabase (PostgreSQL)
├─ ingest_jobs (pipeline state)
├─ crawled_pages (raw content)
└─ chunks (processed embeddings)
```

---

## 📋 Features

### ✅ Completed
- ✓ **Smart crawling** with LLM agent for strategy selection
- ✓ **8-type page classifier** (docs, product, blog, FAQ, pricing, legal, support, general)
- ✓ **Intelligent chunking** with Gemini structured output + validation
- ✓ **Hybrid vector search** (dense Gemini + sparse SPLADE)
- ✓ **Query rewriting** for better semantic matching
- ✓ **CrossEncoder reranking** for result quality
- ✓ **FastAPI REST API** with streaming responses
- ✓ **Production logging** with LangSmith tracing on all 12+ pipeline stages
- ✓ **Supabase integration** for state tracking
- ✓ **Resume capability** — pipeline checkpointing for fault tolerance
- ✓ **React frontend** with Tailwind CSS

### 📝 In Progress
- [ ] Advanced filtering (by content type, date, etc.)
- [ ] Multi-language support
- [ ] GraphQL API option
- [ ] Docker deployment templates

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Google Gemini Flash |
| **Embeddings** | gemini-embedding-001 (768-dim) + SPLADE |
| **Web Crawling** | Crawl4AI (BFS strategy) |
| **Vector DB** | ChromaDB Cloud |
| **State DB** | Supabase (PostgreSQL) |
| **Backend** | FastAPI 0.136 |
| **Frontend** | React 18 + Vite + Tailwind |
| **Reranking** | sentence-transformers CrossEncoder |
| **Monitoring** | LangSmith + custom logger |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Retry Logic** | Tenacity |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 16+ (for frontend)
- Supabase account + API credentials
- ChromaDB Cloud account + API key
- Google API key (Gemini)
- LangSmith API key (optional, for tracing)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/yourusername/website-intelligence-ai.git
cd website-intelligence-ai

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configure Environment Variables

Create `.env` file in project root:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# ChromaDB Cloud
CHROMADB_TENANT=your_tenant
CHROMADB_API_KEY=your_chromadb_key

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key

# LangSmith (optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT=website-intelligence-ai

# FastAPI
FASTAPI_HOST=localhost
FASTAPI_PORT=8000

# Frontend
VITE_API_URL=http://localhost:8000
```

### 3. Initialize Database

```bash
# Supabase tables are auto-created on first ingest call
# Or manually run SQL in Supabase console:
python -c "from utils.database import db; print('DB connected')"
```

---

## 🚀 Running the Application

### Backend (FastAPI Server)

```bash
source .venv/bin/activate

# Development with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API available at: `http://localhost:8000`
Docs available at: `http://localhost:8000/docs` (Swagger UI)

### Frontend (React Dev Server)

```bash
cd frontend
npm run dev
```

Frontend available at: `http://localhost:5173`

---

## 📡 API Reference

### Health Check
```bash
GET /health
```

### Ingest Website

**Start ingestion (runs in background):**
```bash
POST /ingest/
Content-Type: application/json

{
  "url": "https://example.com"
}

# Response:
{
  "status": "started",
  "message": "Ingestion started in background",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Check ingestion status:**
```bash
GET /ingest/{job_id}

# Response:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "company_id": "example.com",
  "start_url": "https://example.com",
  "status": "completed",
  "pages_crawled": 42,
  "chunks_created": 156,
  "company_type": "docs",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Resume failed ingest:**
```bash
POST /ingest/resume/{job_id}

# Response:
{
  "status": "resuming",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Query Website

**Stream query results:**
```bash
POST /query/stream
Content-Type: application/json

{
  "query": "How do I reset my password?",
  "url": "https://example.com",
  "messages": [
    {"role": "user", "content": "What's your pricing?"},
    {"role": "assistant", "content": "We have three plans..."}
  ]
}

# Response: Server-Sent Events (SSE)
data: {"stage": "rewriting", "query": "password reset instructions"}
data: {"stage": "retrieving", "k": 15}
data: {"stage": "reranking", "results": 12}
data: {"stage": "generating", "has_answer": true}
data: {"answer": "You can reset your password by...", "sources": [...]}
```

---

## 💾 Database Schema

### `ingest_jobs`
Tracks pipeline execution state for each URL

```sql
id                 UUID PRIMARY KEY
company_id         TEXT NOT NULL  -- Extracted from URL
start_url          TEXT NOT NULL
status             TEXT           -- in_progress, completed, failed, chunking_failed, embedding_failed
pages_crawled      INT DEFAULT 0
chunks_created     INT DEFAULT 0
company_type       TEXT DEFAULT 'default'  -- Type detected during crawl
created_at         TIMESTAMP DEFAULT NOW()
updated_at         TIMESTAMP
```

### `crawled_pages`
Raw content from website crawl

```sql
id                 UUID PRIMARY KEY
job_id             UUID REFERENCES ingest_jobs
company_id         TEXT NOT NULL
url                TEXT NOT NULL
content            TEXT           -- Raw HTML/Markdown
content_hash       TEXT           -- MD5 hash for deduplication
metadata           JSONB          -- {title, description, depth, parent_url}
crawl_config       JSONB          -- {site_type, max_depth, max_pages, pruning_threshold}
status             TEXT           -- chunking_pending, chunked
created_at         TIMESTAMP
```

### `chunks`
Processed embeddings ready for retrieval

```sql
id                 UUID PRIMARY KEY
job_id             UUID REFERENCES ingest_jobs
page_id            UUID REFERENCES crawled_pages
company_id         TEXT NOT NULL
text               TEXT NOT NULL -- Chunk content
metadata           JSONB          -- {source_url, section_title, keywords, entities, content_type}
status             TEXT           -- ready_for_embedding, embedded
company_type       TEXT           -- For config selection
created_at         TIMESTAMP
```

---

## 🔄 Pipeline Execution Flow

### Ingest Pipeline (Crawl → Chunk → Embed)

```
1. run_ingest()
   ├─ create_ingest_job()          → Create DB record, get job_id
   ├─ run_crawl()                  → Stream pages from crawler
   │  ├─ plan_crawl()              → LLM decides crawl strategy (depth, pages, pruning)
   │  ├─ crawl_url()               → Crawl4AI BFS + checkpoint recovery
   │  └─ save_pages_to_db()        → Store raw content + hash check (dedup)
   │
   ├─ run_chunking()               → Process all crawled pages
   │  ├─ create_chunk_plan()       → Classify page type (8 types)
   │  │  ├─ heuristic_router()     → Fast URL/title pattern matching
   │  │  └─ llm_planner()          → LLM analysis for complex pages
   │  ├─ process_record()          → Chunk single page
   │  │  ├─ build_chunk_prompt()   → Dynamic instructions
   │  │  ├─ call_llm_chunker()     → Gemini structured output
   │  │  ├─ validate_chunks()      → Remove empty/duplicates, split oversized
   │  │  └─ fallback_chunk_page()  → Recursive splitting if LLM fails
   │  └─ [Insert chunks to DB with status "ready_for_embedding"]
   │
   └─ run_embedding()              → Generate embeddings
      ├─ Get all chunks with status "ready_for_embedding"
      ├─ safe_upsert()             → ChromaDB upsert with retry
      └─ Update chunk status → "embedded"

2. resume_pipeline()               → For fault recovery
   └─ Handles: in_progress, chunking_failed, embedding_failed
```

### Query Pipeline (Retrieve → Rerank → Generate)

```
1. retrieve()
   ├─ rewrite_query()              → LLM: "How reset password?" → "password reset instructions"
   ├─ build_filter()               → Extract metadata filters from query
   ├─ hybrid_query()               → Search both dense + sparse embeddings
   │  └─ RRF rank fusion (weights configurable per company_type)
   ├─ parse_results()              → Normalize ChromaDB response
   ├─ remove_duplicates()          → Dedup via content hash
   ├─ rerank()                     → CrossEncoder scoring
   └─ format_context()             → Build source citations

2. generate()
   ├─ has_sufficient_context()     → Check if > 50 chars
   ├─ build_prompt()               → System + history + context + query
   └─ Invoke Gemini → Answer + sources
```

---

## 🔧 Configuration

### Company Type Configs
Each company type has different vector search weights:

```python
# store.py - build_company_config()
"docs"       → dense_weight: 0.6, sparse_weight: 0.4, space: "cosine"
"support"    → dense_weight: 0.7, sparse_weight: 0.3, space: "cosine"
"ecommerce"  → dense_weight: 0.5, sparse_weight: 0.5, space: "l2"
"blog"       → dense_weight: 0.8, sparse_weight: 0.2, space: "cosine"
"default"    → dense_weight: 0.7, sparse_weight: 0.3, space: "cosine"
```

### Crawl Strategy Selection (in crawler.py)
- **Documentation sites** (`/docs`, `/guides`): depth=3, pages=80
- **E-commerce sites**: depth=2, pages=50, JS-heavy
- **Blogs**: depth=2, pages=30
- **Support/Help centers**: depth=2, pages=60
- **Unknown sites**: depth=1, pages=20

### Page Type Routing (in chunk_planner.py)
- **Docs** → section_based chunking, preserve code blocks
- **Product/Features** → feature_spec style, preserve tables
- **Pricing** → pricing_blocks (never split across tiers)
- **Blog** → semantic_flow, preserve chronology
- **FAQ** → qa_pairs (keep Q&A together)
- **Legal/Terms** → clause_exact, preserve numbering
- **Support** → thematic with metadata focus

---

## 📊 Monitoring & Debugging

### LangSmith Traces
Every function is `@traceable`, sends execution data to LangSmith:

```bash
# All traceable functions:
- plan_crawl, crawl_url
- heuristic_router, llm_planner, create_chunk_plan
- process_record, validate_chunks, call_llm_chunker
- run_embedding, safe_upsert
- create_ingest_job, save_pages_to_db, run_crawl, run_chunking, run_ingest, resume_pipeline
- rewrite_query, build_filter, hybrid_query, rerank, retrieve
- generate

# View in LangSmith dashboard: https://smith.langchain.com
```

### Logging
Production-grade logger in `utils/logger.py`:

```bash
# Logs written to:
logs/
├─ CRAWLER.log
├─ CHUNKER.log
├─ RETRIEVER.log
├─ INGEST_PIPELINE.log
└─ ...

# Default level: INFO (set via environment)
```

### Database Queries
Monitor pipeline progress:

```python
from utils.database import db

# Check ingestion status
job = db.table("ingest_jobs")\
    .select("*")\
    .eq("company_id", "example.com")\
    .order("created_at", desc=True)\
    .limit(1)\
    .execute()

# Get pages crawled for a job
pages = db.table("crawled_pages")\
    .select("id, url, content_hash")\
    .eq("job_id", job_id)\
    .execute()

# Get chunks for search
chunks = db.table("chunks")\
    .select("text, metadata")\
    .eq("company_id", "example.com")\
    .eq("status", "embedded")\
    .execute()
```

---

## 🚢 Deployment

### Docker (Coming Soon)
```dockerfile
# Dockerfile included in repo
docker build -t website-intelligence-ai .
docker run -p 8000:8000 --env-file .env website-intelligence-ai
```

### Railway Deployment
```bash
# Connect repo to Railway
railway link
railway up

# Set environment variables in Railway dashboard
# Deploy: git push
```

### Vercel (Frontend Only)
```bash
cd frontend
npm run build
# Deploy `dist/` folder to Vercel
```

---

## 🤝 Contributing

Contributions welcome! This is an AI engineer learning project.

**Development workflow:**
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with clear commit messages
3. Test locally: `pytest test.py`
4. Push: `git push origin feature/your-feature`
5. Create PR with description

**Code style:**
- Python: Black formatter, type hints
- Frontend: ESLint + Prettier (configured in repo)
- Docstrings: Google-style (see existing code)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file

---

## 🙋 Support

**Issues?** Check these first:
1. Environment variables in `.env` are set correctly
2. Supabase tables exist (auto-created on first ingest)
3. API keys are valid (test with `curl localhost:8000/health`)
4. Check logs in `logs/` directory

**Questions?** Open an issue on GitHub or check the [AGENTS.md](AGENTS.md) for project context.

---

## 📚 Additional Resources

- **Architecture**: See [AGENTS.md](AGENTS.md)
- **API Docs**: http://localhost:8000/docs (after running backend)
- **LangSmith Dashboard**: https://smith.langchain.com (requires API key)
- **Supabase Console**: https://app.supabase.com (manage database)
- **ChromaDB Docs**: https://docs.trychroma.com

---

**Built with ❤️ by Neel Mahyavanshi — AI Engineer in Training**
