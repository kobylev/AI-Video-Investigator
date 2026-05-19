import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiReasoner:
    """Agent 2: Deep semantic verification using Gemini 1.5 Pro."""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def verify_event(self, frame: Image.Image, query: str) -> bool:
        """
        Binary classification: Is the event described in the query present in this frame?
        """
        prompt = f"""
        SYSTEM PROMPT: You are an expert forensic video investigator.
        TASK: Analyze the provided frame and determine if the following event is occurring.
        EVENT TO VERIFY: "{query}"
        
        RESPONSE FORMAT: 
        Respond ONLY with 'TRUE' if the event is clearly occurring, or 'FALSE' if it is not.
        Do not provide any other text.
        """
        
        response = self.model.generate_content([prompt, frame])
        result = response.text.strip().upper()
        
        # Track usage via performance tracker in main loop
        return "TRUE" in result, response
