import sys
import json
sys.path.append("src")
from PIL import Image
from reasoner.claude_engine import ClaudeReasoner, ReasonerVerdict

def main():
    reasoner = ClaudeReasoner()
    verdict = reasoner.verify_event(Image.open("sample_frame.jpg"), "a dog chasing a person")
    assert isinstance(verdict, ReasonerVerdict), "verdict has wrong type"
    assert isinstance(verdict.event_detected, bool), "event_detected not bool"
    assert 0.0 <= verdict.confidence_score <= 1.0, "confidence_score out of range"
    assert isinstance(verdict.reasoning, str) and verdict.reasoning, "reasoning empty"

    print(json.dumps({
        "event_detected":   verdict.event_detected,
        "confidence_score": verdict.confidence_score,
        "reasoning":        verdict.reasoning[:200],
        "is_verified":      verdict.is_verified,
    }, indent=2))

if __name__ == "__main__":
    main()
