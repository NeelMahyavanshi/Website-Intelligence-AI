from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

print(os.getenv("OPENROUTER_API_KEY"))
print(os.getenv("OPENROUTER_MODEL"))

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  model="openrouter/free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
