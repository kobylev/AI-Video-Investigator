from typing import List, Optional
from pydantic import BaseModel, Field

class ForensicVerdict(BaseModel):
    """Structured output contract for Claude forensic analysis."""
    chain_of_thought: str = Field(
        ..., 
        description="Step-by-step visual reasoning identifying specific objects, motions, or behaviors."
    )
    is_event_present: bool = Field(
        ..., 
        description="Final binary verdict on whether the queried event is occurring."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Model's confidence in the verdict."
    )
    forensic_summary: str = Field(
        ..., 
        max_length=280,
        description="A concise summary of the evidence for the final investigation report."
    )

class PromptManager:
    """
    Manages domain-specific security prompt templates with CoT enforcement.
    """
    
    TEMPLATES = {
        "unauthorized_entry": (
            "Analyze the frame for tailgating or unauthorized entry. "
            "1. Identify the secure portal (door/gate). "
            "2. Count the persons/vehicles passing. "
            "3. Look for 'piggybacking' behavior where a second entity follows without credentialing."
        ),
        "crowd_behavior": (
            "Analyze the frame for abnormal crowd behavior or loitering. "
            "1. Estimate crowd density. "
            "2. Identify rapid movements or aggressive posturing. "
            "3. Note individuals remaining stationary in high-traffic zones for extended periods."
        ),
        "traffic_violation": (
            "Analyze the dashcam footage for traffic violations. "
            "1. Identify lane markings and vehicle proximities. "
            "2. Look for 'cutting off' (unsafe lane changes) or sudden braking. "
            "3. Check for proximity to intersections or traffic control devices."
        ),
        "generic": (
            "Analyze the frame for the specific user-defined event: '{query}'. "
            "Perform a step-by-step visual audit of the scene before reaching a conclusion."
        )
    }

    @classmethod
    def get_prompt(cls, query: str, category: str = "generic") -> str:
        template = cls.TEMPLATES.get(category, cls.TEMPLATES["generic"])
        
        base_instruction = (
            "You are a Senior Forensic Video Analyst. Your task is to verify an event "
            "in the provided surveillance frame. You MUST use Chain-of-Thought reasoning.\n\n"
            f"SPECIFIC INSTRUCTIONS: {template.format(query=query)}\n\n"
            "OUTPUT FORMAT: You must return a valid JSON object matching the requested schema."
        )
        return base_instruction
