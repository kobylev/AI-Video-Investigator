import os
from dotenv import load_dotenv
import anthropic

load_dotenv(override=True)
key = os.getenv("ANTHROPIC_API_KEY")
print("Key:", repr(key[:15] + "..." if key else None))

try:
    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print("API Response:", response.content)
    print("Success!")
except Exception as e:
    print("Failed to call Claude API:", e)

