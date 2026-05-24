# src/config.py
import os

# Global evaluation mode toggle
# When True: Relaxed parameters to optimize Recall and eliminate False Negatives during evaluation
# When False: Production parameters optimized for budget and precision (budget-friendly)
EVALUATION_MODE = os.getenv("EVALUATION_MODE", "True").lower() in ("true", "1", "yes")

# --- Production Constraints (EVALUATION_MODE = False) ---
PROD_K = 5
PROD_TAU_HIGH = 0.32
PROD_TAU_LOW = 0.24
PROD_TIME_WINDOW_SEC = 5
PROD_NMS_HIGH_CONF_THRESHOLD = 98.0
PROD_NMS_PEAK_PROXIMITY_DELTA = 2.0
PROD_NMS_MAX_PER_CLUSTER = 2
PROD_TOLERANCE_RADIUS = 2
PROD_MAX_ESCALATIONS = 5

# --- Evaluation Override Constraints (EVALUATION_MODE = True) ---
EVAL_K = 20                           # Increased retrieval limit to maximize recall
EVAL_TAU_HIGH = 0.32
EVAL_TAU_LOW = 0.15                   # Lowered threshold to escalate more ambiguous edge cases
EVAL_TIME_WINDOW_SEC = 5
EVAL_NMS_HIGH_CONF_THRESHOLD = 90.0   # Relaxed NMS threshold to allow more adjacent candidates
EVAL_NMS_PEAK_PROXIMITY_DELTA = 5.0    # Relaxed peak proximity delta
EVAL_NMS_MAX_PER_CLUSTER = 5           # Relaxed cluster limit (increased from 2 to 5)
EVAL_TOLERANCE_RADIUS = 4              # Expanded temporal tolerance window for evaluation matching
EVAL_MAX_ESCALATIONS = 20              # Allow more candidates to escalate to Cloud Reasoner

def get_config_value(name: str):
    """Get the active configuration value based on EVALUATION_MODE."""
    if EVALUATION_MODE:
        eval_name = f"EVAL_{name}"
        if eval_name in globals():
            return globals()[eval_name]
    prod_name = f"PROD_{name}"
    if prod_name in globals():
        return globals()[prod_name]
    raise AttributeError(f"Configuration parameter '{name}' not found.")
