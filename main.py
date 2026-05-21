import os
import sys
import logging
from PIL import Image
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from router.core import BudgetAwareRouter
from reasoner.claude_engine import ClaudeReasoner, ReasonerVerdict
from reasoner.prompt_manager import PromptManager

def run_mock_pipeline():
    """
    Demonstrates the WP5 Dual-Agent Cascade with Claude Haiku 4.5.
    """
    load_dotenv()
    
    # 1. Setup Components
    router = BudgetAwareRouter(tau_high=0.35, tau_low=0.25, max_escalations=2)
    reasoner = ClaudeReasoner() # Expects ANTHROPIC_API_KEY in .env
    
    # 2. Mock CLIP Candidates (usually from FAISS in WP4)
    dummy_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    
    mock_candidates = [
        {"frame": dummy_img, "timestamp": 12.5, "score": 0.38}, # High Match
        {"frame": dummy_img, "timestamp": 45.2, "score": 0.29}, # Ambiguous -> Escalate
        {"frame": dummy_img, "timestamp": 88.0, "score": 0.22}, # Low Match -> Drop
    ]
    
    print(f"--- Stage 1: Confidence-Gated Routing ---")
    accepted, ambiguous = router.route_candidates(mock_candidates)
    
    print(f"Accepted (Direct): {len(accepted)}")
    print(f"Ambiguous (Escalated): {len(ambiguous)}")
    
    # 3. Stage 2: Reasoner Escalation
    query = "unauthorized vehicle tailgating at the gate"
    
    print(f"\n--- Stage 2: Claude Reasoner Escalation (Haiku 4.5) ---")
    final_results = []
    
    # Add auto-accepted items
    for item in accepted:
        item['is_verified'] = True
        item['forensic_summary'] = "Auto-accepted by CLIP (High Confidence)"
        final_results.append(item)
    
    # Process Ambiguous items via Claude
    if ambiguous:
        for item in ambiguous:
            logger.info(f"[*] Reasoner processing frame at {item['timestamp']}s...")
            verdict: ReasonerVerdict = reasoner.verify_event(item['frame'], query)
            
            item['is_verified'] = verdict.is_verified
            item['forensic_summary'] = verdict.reasoning
            item['confidence_score'] = verdict.confidence_score
            final_results.append(item)
    
    # 4. Final Report
    print(f"\n--- Final Investigation Report ---")
    for res in sorted(final_results, key=lambda x: x['timestamp']):
        status = "✅ VERIFIED" if res.get('is_verified') else "❌ REJECTED"
        summary = res.get('forensic_summary')
        print(f"[{res['timestamp']}s] {status} - {summary}")

    report = router.get_budget_report()
    print(f"\nBudget Audit: Estimated Spend ${report['total_estimated_spend_usd']:.4f}")

if __name__ == "__main__":
    # Note: To run this, you need a valid ANTHROPIC_API_KEY in your .env
    try:
        run_mock_pipeline()
    except Exception as e:
        print(f"Pipeline failed: {e}")
