import time
from dataclasses import dataclass

@dataclass
class ExecutionMetrics:
    """Audit log for academic defense: tracks latency and token costs."""
    start_time: float = 0
    end_time: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoner_invoked: bool = False
    
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        
    @property
    def latency(self):
        return self.end_time - self.start_time
    
    @property
    def estimated_cost_usd(self):
        # Pricing based on Gemini 1.5 Pro: $3.50/1M input, $10.50/1M output
        input_cost = (self.input_tokens / 1_000_000) * 3.50
        output_cost = (self.output_tokens / 1_000_000) * 10.50
        return input_cost + output_cost

class PerformanceTracker:
    def __init__(self):
        self.metrics = ExecutionMetrics()

    def log_api_usage(self, response):
        """Extracts token usage from Gemini or Claude responses."""
        # 1. Handle Gemini (google-genai)
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            self.metrics.input_tokens += getattr(usage, 'prompt_token_count', 0)
            self.metrics.output_tokens += getattr(usage, 'candidates_token_count', 0)
        
        # 2. Handle Claude (anthropic)
        usage_claude = getattr(response, 'usage', None)
        if usage_claude:
            self.metrics.input_tokens += getattr(usage_claude, 'input_tokens', 0)
            self.metrics.output_tokens += getattr(usage_claude, 'output_tokens', 0)
            
        self.metrics.reasoner_invoked = True
