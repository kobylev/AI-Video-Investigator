import anthropic
from PIL import Image
import os
import io
import base64
from dotenv import load_dotenv

load_dotenv()

class ClaudeReasoner:
    """Agent 2: Deep semantic verification using Anthropic Claude Haiku 4.5."""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = "claude-haiku-4-5-20251001"
        
    def verify_event(self, frame: Image.Image, query: str) -> bool:
        """
        Binary classification using Claude Haiku 4.5.
        """
        import time
        
        # Convert PIL Image to base64 for Claude
        img_byte_arr = io.BytesIO()
        # Resize to maintain forensic detail while keeping token costs low
        frame_copy = frame.copy()
        frame_copy.thumbnail((512, 512))
        frame_copy.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = f"""
        SYSTEM PROMPT: You are an expert forensic video investigator.
        TASK: Analyze the provided frame and determine if the following event is occurring.
        EVENT TO VERIFY: "{query}"
        
        RESPONSE FORMAT: 
        Respond ONLY with 'TRUE' if the event is clearly occurring, or 'FALSE' if it is not.
        Do not provide any other text.
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=10,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_base64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ],
                        }
                    ]
                )
                
                result = response.content[0].text.strip().upper()
                return "TRUE" in result, response
                
            except Exception as e:
                # Handle potential rate limits (OverloadedError or RateLimitError)
                if ("overloaded" in str(e).lower() or "rate_limit" in str(e).lower()) and attempt < max_retries - 1:
                    print(f"\n[!] Claude API issue. Waiting 30s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(30)
                    continue
                raise e
