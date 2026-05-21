"""
[DEPRECATED] Initial implementation using Google Gemini.
Standardized Reasoning Agent is now Anthropic Claude Haiku 4.5 (see src/reasoner/claude_engine.py).
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai
from pydantic import ValidationError

from .prompt_manager import ForensicVerdict

logger = logging.getLogger(__name__)

class GeminiReasoner:
    """
    Scalable Wrapper for Gemini 1.5 Pro. (Historical Reference Only)
    """

    def __init__(
        self, 
        model_name: str = "gemini-1.5-pro-latest",
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        
        # Generation config for JSON mode
        self.generation_config = {
            "temperature": 0.1,  # Low temp for deterministic forensic output
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        }

    def verify_frame(
        self, 
        frame: Image.Image, 
        system_prompt: str
    ) -> ForensicVerdict:
        """
        Sends a single frame and prompt to Gemini and parses the structured result.
        """
        try:
            # Prepare content list (multimodal)
            content = [system_prompt, frame]
            
            response = self.model.generate_content(
                content,
                generation_config=self.generation_config
            )
            
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            
            # Parse and validate via Pydantic
            data = json.loads(response.text)
            return ForensicVerdict(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Gemini output did not match ForensicVerdict schema: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise

    def batch_verify(
        self, 
        escalated_items: List[Dict[str, Any]], 
        system_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Processes a list of escalated candidates from the Router.
        """
        results = []
        for item in escalated_items:
            logger.info(f"[*] Reasoner processing frame at {item['timestamp']}s...")
            verdict = self.verify_frame(item['frame'], system_prompt)
            
            # Merge verdict into item data
            item['verdict'] = verdict.dict()
            item['is_verified'] = verdict.is_event_present
            results.append(item)
            
        return results
