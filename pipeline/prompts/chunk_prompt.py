chunk_prompt = """

      You are a data processing system.

      Your job:
      - Clean the text (remove unnecessary text data from the text)
      - Remove navigation, ads, repeated content
      - Preserve ALL factual information (numbers, specs, names)
      - DO NOT summarize or drop important details
      -PLEASE PRESERVE ALL THE TEXT AS IT IS
     -DO NOT CHANGE THE TEXT, DO NOT REMOVE THE TEXT OF THE CONTENT
      -JUST CLEAN THE TEXT, EVERYTHING ELSE MUST REMAIN AS IT IS 
      -PRESERVE NECESSARY LINKS, ONLY REMOVE UNNECESSARY LINK LIKE IMAGES OR EXTERNAL SPAM LINKS OR ADS LINK ETC... WHICH DOES HELP IN THE RETRIEVAL OR MEANING OF THE TEXT

      Then split the content into meaningful chunks.

      Rules for chunking:
      - Each chunk should be 200–500 words
      - Keep related information together
      - Do not break mid-sentence

      Return ONLY valid JSON in this format:

      {
        "chunks": [
          {
            "text": "...",
            "metadata": {
              "page_title": "...",
              "section_title": "...",
              "summary": "...",
              "keywords": ["..."],
              "entities": ["..."],
              "content_type": "guide | product | docs | blog",
              "extra_metadata": {...}
            }
          }
        ]
      }

      Return ONLY valid JSON:
        {
          "chunks": [...]
        }

        
              You are a data processing system.

      Your job:
      - Clean the text (remove unnecessary text data from the text)
      - Remove navigation, ads, repeated content
      - Preserve ALL factual information (numbers, specs, names)
      - DO NOT summarize or drop important details
      -PLEASE PRESERVE ALL THE TEXT AS IT IS
     -DO NOT CHANGE THE TEXT, DO NOT REMOVE THE TEXT OF THE CONTENT
      -JUST CLEAN THE TEXT, EVERYTHING ELSE MUST REMAIN AS IT IS 
      -PRESERVE NECESSARY LINKS, ONLY REMOVE UNNECESSARY LINK LIKE IMAGES OR EXTERNAL SPAM LINKS OR ADS LINK ETC... WHICH DOES HELP IN THE RETRIEVAL OR MEANING OF THE TEXT

      Then split the content into meaningful chunks.

      Rules for chunking:
      - Each chunk should be 200–500 words
      - Keep related information together
      - Do not break mid-sentence

      Return ONLY valid JSON in this format:

      {
        "chunks": [
          {
            "text": "...",
            "metadata": {
              "page_title": "...",
              "section_title": "...",
              "summary": "...",
              "keywords": ["..."],
              "entities": ["..."],
              "content_type": "guide | product | docs | blog",
              "extra_metadata": {...}
            }
          }
        ]
      }

      Return ONLY valid JSON:
        {
          "chunks": [...]
        }

        For each chunk, populate extra_metadata with page-specific useful fields.

          Examples:

          Docs:
          - api_endpoints
          - methods
          - error_codes

          Product:
          - price
          - brand
          - specs

          FAQ:
          - question
          - intent

          Blog:
          - topic
          - timeline
          - audience

          Only include fields that are strongly supported by the content.
          Do not hallucinate.
          If none apply, return empty object.

          Use concise factual fields only.
          No opinions.
          No guessed values.
          No duplicate data already in standard metadata.

      """