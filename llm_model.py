from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)


llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    temperature=0,
    default_headers={
        "HTTP-Referer": "https://github.com/neelmahyavanshi/website-intelligence-ai",
        "X-Title": "Website Intelligence AI"
    },
    extra_body={
        "reasoning": {"effort": "none"}
    }
)

# llm = ChatGoogleGenerativeAI(
#     model=os.getenv("GOOGLE_MODEL"),
#     temperature=0,
# )