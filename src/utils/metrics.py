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
        """Extracts token usage from Google Generative AI response."""
        if hasattr(response, 'usage_metadata'):
            self.metrics.input_tokens += response.usage_metadata.prompt_token_count
            self.metrics.output_tokens += response.usage_metadata.candidates_token_count
        self.metrics.reasoner_invoked = True
