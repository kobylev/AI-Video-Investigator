import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from PIL import Image

class BudgetAwareRouter:
    """
    Implements Confidence-Gated Routing (WP3 Core Logic).
    Gathers frames from Retriever and decides whether to accept, escalate, or discard.
    """
    
    def __init__(self, tau_high: float = 0.80, tau_low: float = 0.65):
        """
        :param tau_high: Scores above this are accepted immediately.
        :param tau_low: Scores below this are discarded.
        """
        self.tau_high = tau_high
        self.tau_low = tau_low

    def route_request(self, frame_scores: List[Tuple[Image.Image, float, float]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Routes based on the 'Match %' (probability score).
        Returns list of frames to be accepted immediately and frames to be escalated.
        """
        accepted = []
        ambiguous = []
        
        for frame, timestamp, probability in frame_scores:
            # 1. HIGH MATCH (Above tau_high): 
            # CLIP is highly certain. Immediate return (Token Economy: 100% saved)
            if probability >= self.tau_high:
                accepted.append({
                    "frame": frame,
                    "timestamp": timestamp,
                    "score": probability,
                    "routing": "IMMEDIATE_MATCH"
                })
                
            # 2. AMBIGUOUS BAND (Between tau_low and tau_high):
            # Escalation to Reasoner (Gemini) for high-order verification.
            elif self.tau_low <= probability < self.tau_high:
                ambiguous.append({
                    "frame": frame,
                    "timestamp": timestamp,
                    "score": probability,
                    "routing": "ESCALATED"
                })
                
            # 3. LOW MATCH (Below tau_low):
            # Discarded - CLIP's confidence is too low to justify the cost of reasoning.
                
        return accepted, ambiguous
