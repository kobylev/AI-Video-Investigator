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
        self.model_id = 'gemini-2.0-flash'
        
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
        # Resize to a reasonable forensic resolution (512px) to save token quota
        # while maintaining enough detail for the Reasoner.
        frame_copy = frame.copy()
        frame_copy.thumbnail((512, 512))
        frame_copy.save(img_byte_arr, format='PNG')
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
                # Catch the specific 403 Permission Denied error
                if "403" in str(e):
                    print(f"\n[!] CRITICAL ERROR: API Key Permission Denied (403).")
                    print("[*] FIX: Go to https://aistudio.google.com/ and create a NEW API Key.")
                    print("[*] Alternative: Enable 'Generative Language API' in Google Cloud Console.")
                    return False, None

                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"\n[!] Rate limit hit. Waiting 60s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(60)
                    continue
                raise e

