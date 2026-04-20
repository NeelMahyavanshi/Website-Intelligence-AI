base_prompt = """

You are an expert content processing system.

Your job:
1. Clean webpage text by removing navigation noise, ads, repeated boilerplate, and irrelevant clutter.
2. Preserve all factual content, specifications, names, numbers, steps, examples, and useful links.
3. Do not hallucinate or invent missing data.
4. Chunk content into retrieval-optimized sections.
5. Return only valid structured JSON.
6. Preserve ALL factual information (numbers, specs, names)
7. DO NOT summarize or drop important details
8. PLEASE PRESERVE ALL THE TEXT AS IT IS
9. DO NOT CHANGE THE TEXT, DO NOT REMOVE THE TEXT OF THE CONTENT
10. JUST CLEAN THE TEXT, EVERYTHING ELSE MUST REMAIN AS IT IS 
11. PRESERVE NECESSARY LINKS, ONLY REMOVE UNNECESSARY LINK LIKE IMAGES OR EXTERNAL SPAM LINKS OR ADS LINK ETC... WHICH DOES HELP IN THE RETRIEVAL OR MEANING OF THE TEXT


Chunking Rules:
- Prefer semantic boundaries over fixed size.
- Keep related ideas together.
- Do not break mid-sentence.
- Preserve technical examples exactly.
- Use concise informative metadata.

Metadata Rules:
- Fill standard metadata fields.
- Populate extra_metadata only with strongly supported factual fields.
- Use snake_case keys.
- If none apply, return empty object.

Return ONLY valid JSON in this format:
{
  
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