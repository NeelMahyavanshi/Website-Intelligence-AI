base_prompt = f"""
You are an expert content chunking system.

Goal:
Transform webpage content into retrieval-optimized chunks.

Global Rules:
- Preserve factual accuracy
- Do not hallucinate
- Do not remove important details
- Target chunk size: {plan.target_chunk_words} words
- Return only valid structured JSON
- Use snake_case keys in extra_metadata
"""


templates = {

        "docs": """
Page Type: Technical Documentation

Instructions:
- Chunk by headings / sections
- Preserve code blocks exactly
- Keep setup steps together
- Keep API examples intact
- Extract api_endpoints, methods, error_codes
""",

        "product": """
Page Type: Product

Instructions:
- Keep product specs together
- Preserve dimensions, variants, pricing
- Keep feature comparisons intact
- Extract price, brand, specs, sku
""",

        "blog": """
Page Type: Blog

Instructions:
- Preserve narrative flow
- Chunk by topic transitions
- Keep chronology intact
- Extract themes, audience, timeline
""",

        "faq": """
Page Type: FAQ

Instructions:
- One question-answer pair per chunk
- Preserve exact wording of question
- Extract intent, category
""",

        "support": """
Page Type: Support

Instructions:
- Chunk by troubleshooting step groups
- Preserve symptoms and resolutions
- Extract issue_type, product_area
""",

        "pricing": """
Page Type: Pricing

Instructions:
- Keep pricing tiers separate
- Preserve feature comparisons
- Extract plans, prices, billing_cycle
""",

        "legal": """
Page Type: Legal

Instructions:
- Preserve clause wording exactly
- Chunk by sections
- Extract clause_type, jurisdiction
""",

        "general": """
Page Type: General

Instructions:
- Chunk semantically
- Keep related ideas together
""",

        "other": """
Page Type: Other

Instructions:
- Chunk semantically
- Preserve useful information
"""
    }