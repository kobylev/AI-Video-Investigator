import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not found.")
    exit(1)

client = Anthropic(api_key=api_key)

print("Listing available Anthropic models (capability check)...")
# Note: Anthropic doesn't have a direct 'list models' API like Google, 
# so we manually verify the target model.
target_model = "claude-haiku-4-5-20251001"
print(f"Target Reasoner: {target_model}")
