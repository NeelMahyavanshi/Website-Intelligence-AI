from pipeline.chunker import process_record

samples = [

{
    "url": "https://example.com/docs/auth",
    "content": """
# Authentication Guide

Use POST /login to receive JWT token.

Headers:
Authorization: Bearer <token>

Error Codes:
401 Unauthorized
403 Forbidden

Example Python code:

requests.post("/login")
""",
    "metadata": {
        "title": "Authentication Docs",
        "description": "JWT authentication guide"
    },
    "timestamp": "2026-01-01"
},

{
    "url": "https://example.com/pricing",
    "content": """
Pricing Plans

Starter - $19/month
10 users
Email support

Pro - $49/month
Unlimited users
Priority support
Analytics included
""",
    "metadata": {
        "title": "Pricing"
    },
    "timestamp": "2026-01-01"
},

{
    "url": "https://example.com/blog/ai-agents",
    "content": """
How AI Agents Are Changing Work

AI agents can automate repetitive workflows.
In 2025 many startups adopted multi-agent systems.

Teams now use agents for support, coding and research.
""",
    "metadata": {
        "title": "AI Agents Blog"
    },
    "timestamp": "2026-01-01"
}

]

for sample in samples:

    print("=" * 80)
    print("URL:", sample["url"])

    chunks = process_record(sample)

    print("Chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(chunk["text"][:300])
        print(chunk["metadata"])