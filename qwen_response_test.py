
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "user",
            "content": "What is 12 + 7?"
        }
    ],
    reasoning_effort="none"
)

print(response.choices[0].message.content)
