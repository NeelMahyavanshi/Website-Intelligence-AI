from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# llm = ChatOpenAI(
#     model=os.getenv("OPENROUTER_MODEL"),
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
#     temperature=0,
# )

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL"),
    temperature=0,
)