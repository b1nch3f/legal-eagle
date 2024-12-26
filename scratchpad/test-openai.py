import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment_name = "gpt-4o-mini"

# Send a completion call to generate an answer
print("Sending a test completion job")
start_phrase = "Write a tagline for an ice cream shop. "
response = client.chat.completions.create(
    model=deployment_name, messages=[{"role": "system", "content": f"{start_phrase}"}]
)
print(start_phrase + response.choices[0].message.content)
