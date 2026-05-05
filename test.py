# test_generator.py
from pipeline.retriever import retrieve
from pipeline.generator import generate

company_id = "demo"
company_type = "tech_docs"

queries = [
    "How does login authentication work?",
    "What is the price of pro plan?",
    "Tell me about AI agents",
    "How do I reset my password?",
    "What chunking strategies exist for RAG?",
    "What is the refund policy?",  # should return has_answer=False
]

for query in queries:
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("="*50)

    retrieval_result = retrieve(
        query=query,
        company_id=company_id,
        company_type=company_type,
        k=10
    )

    result = generate(query, retrieval_result)

    print(f"Has Answer: {result['has_answer']}")
    print(f"Sources:    {result['sources']}")
    print(f"\nANSWER:\n{result['answer']}")