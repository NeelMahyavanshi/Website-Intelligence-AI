# test_e2e.py
import asyncio
import requests

BASE_URL = "http://127.0.0.1:8000"
COMPANY_ID = "crawl4ai-docs"
COMPANY_TYPE = "tech_docs"

def test_health():
    print("="*50)
    print("STEP 1 — HEALTH CHECK")
    print("="*50)
    r = requests.get(f"{BASE_URL}/health/")
    print(r.json())

def test_ingest():
    print("\n" + "="*50)
    print("STEP 2 — INGEST CRAWL4AI DOCS")
    print("="*50)
    print("This will take a few minutes...")
    
    r = requests.post(f"{BASE_URL}/ingest/", json={
        "url": "https://docs.crawl4ai.com/",
        "company_id": COMPANY_ID,
        "company_type": COMPANY_TYPE
    }, timeout=600)  # 10 min timeout for crawl
    
    print(r.json())

def test_queries():
    print("\n" + "="*50)
    print("STEP 3 — QUERYING")
    print("="*50)

    queries = [
        "How do I install Crawl4AI?",
        "What is Crawl4AI?",
        "How do I use the Crawl4AI API?",
        "What models are available in Crawl4AI?",
        "How does the Crawl4AI pricing work?",
        "What is the difference between free and paid plans?",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        print("-"*40)
        
        r = requests.post(f"{BASE_URL}/query/", json={
            "query": query,
            "company_id": COMPANY_ID,
            "company_type": COMPANY_TYPE
        }, timeout=60)
        
        result = r.json()
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {result.get('sources', [])}")
        print(f"Has Answer: {result.get('has_answer', False)}")

if __name__ == "__main__":
    test_health()
    test_ingest()   # comment this out after first run
    test_queries()