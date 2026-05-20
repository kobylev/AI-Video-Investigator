from google import genai
from google.genai import types
from PIL import Image
import os
import io
from dotenv import load_dotenv

load_dotenv()

class GeminiReasoner:
    """Agent 2: Deep semantic verification using the new google-genai SDK."""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-1.5-pro'
        
    def verify_event(self, frame: Image.Image, query: str) -> bool:
        """
        Binary classification with automatic retry logic for rate limits.
        """
        import time
        from google.genai import errors
        
        prompt = f"""
        SYSTEM PROMPT: You are an expert forensic video investigator.
        TASK: Analyze the provided frame and determine if the following event is occurring.
        EVENT TO VERIFY: "{query}"
        
        RESPONSE FORMAT: 
        Respond ONLY with 'TRUE' if the event is clearly occurring, or 'FALSE' if it is not.
        Do not provide any other text.
        """
        
        img_byte_arr = io.BytesIO()
        frame.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=img_bytes, mime_type='image/png')
                    ]
                )
                result = response.text.strip().upper()
                return "TRUE" in result, response
                
            except errors.ClientError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"\n[!] Rate limit hit. Waiting 60s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(60) # Wait a full minute for the quota bucket to refill
                    continue
                raise e

