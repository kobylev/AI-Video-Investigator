# Evaluation Metrics & Token Economics Tracker

class TokenEconomics:
    """
    Tracks and auditors API spend across Dual-Agent retrieval pipeline.
    Standardized on Anthropic Claude Haiku 4.5.
    """
    
    # Updated pricing for Claude Haiku 4.5 (approximate)
    PRICE_INPUT_1M = 1.00  # $1.00 / 1M input tokens
    PRICE_OUTPUT_1M = 5.00 # $5.00 / 1M output tokens

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def log_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1_000_000) * self.PRICE_INPUT_1M
        output_cost = (self.total_output_tokens / 1_000_000) * self.PRICE_OUTPUT_1M
        return input_cost + output_cost

    def extract_usage(self, response):
        """Extracts token usage from Anthropic response objects."""
        if hasattr(response, 'usage'):
            self.log_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
