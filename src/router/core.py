from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class RoutingVerdict(BaseModel):
    """Data contract for the router's decision on a specific frame."""
    timestamp: float
    score: float
    action: str  # "ACCEPT", "ESCALATE", or "DROP"
    estimated_cost: float = 0.0

class BudgetAwareRouter:
    """
    Agent 1 Gatekeeper: Implements Confidence-Gated Routing (WP3/WP5).
    Optimizes the balance between retrieval accuracy and API token spend.
    """

    def __init__(
        self, 
        tau_high: float = 0.32, 
        tau_low: float = 0.24, 
        max_escalations: int = 5,
        cost_per_image: float = 0.0003  # Approx cost for Claude Haiku 4.5 image tokens
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.max_escalations = max_escalations
        self.cost_per_image = cost_per_image
        self.total_estimated_spend = 0.0

    def route_candidates(
        self, 
        candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Partitions CLIP results into 'Accepted' (Direct Return) and 'Ambiguous' (Escalate).
        
        :param candidates: List of dicts containing {'frame': Image, 'timestamp': float, 'score': float}
        :return: (accepted_list, escalated_list)
        """
        accepted = []
        ambiguous = []

        # Sort by score descending to ensure we escalate the most promising candidates first
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        for item in sorted_candidates:
            score = item['score']
            
            if score >= self.tau_high:
                # 1. HIGH MATCH: CLIP is confident enough for a direct match.
                item['routing_action'] = "IMMEDIATE_MATCH"
                accepted.append(item)
                logger.info(f"[*] ACCEPTED: Frame at {item['timestamp']}s (Score: {score:.4f})")
            
            elif self.tau_low <= score < self.tau_high:
                # 2. AMBIGUOUS BAND: Needs VLM reasoning.
                if len(ambiguous) < self.max_escalations:
                    item['routing_action'] = "ESCALATED"
                    item['estimated_cost'] = self.cost_per_image
                    ambiguous.append(item)
                    self.total_estimated_spend += self.cost_per_image
                    logger.info(f"[*] ESCALATING: Frame at {item['timestamp']}s (Score: {score:.4f})")
                else:
                    logger.warning(f"[!] BUDGET CAP: Skipping escalation for frame at {item['timestamp']}s")
            
            else:
                # 3. LOW MATCH: Below noise floor.
                logger.debug(f"[-] DROPPED: Frame at {item['timestamp']}s (Score: {score:.4f})")

        return accepted, ambiguous

    def get_budget_report(self) -> Dict[str, Any]:
        return {
            "total_estimated_spend_usd": self.total_estimated_spend,
            "thresholds": {"high": self.tau_high, "low": self.tau_low},
            "max_allowed_escalations": self.max_escalations
        }
