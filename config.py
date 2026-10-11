"""
config.py
---------
Central configuration for the Bloom Taxonomy LLM Classifier.

All configurable values are read from environment variables (via .env)
so that experiments can be reproduced or varied without touching source code.
"""

import os
from dotenv import load_dotenv

# Load variables from .env if it exists (development convenience)
load_dotenv()


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Hugging Face model identifier.  Change this to swap to a different
# open-weight model without modifying any other source file.
MODEL_NAME: str = os.getenv("MODEL_NAME", "google/gemma-3-1b-it")

# Maximum number of new tokens the model may generate per inference call.
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Sampling temperature.  Low values (e.g. 0.1) make output more deterministic.
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

# Nucleus sampling probability threshold.
TOP_P: float = float(os.getenv("TOP_P", "0.9"))

# Device selection: "auto" | "cuda" | "cpu"
DEVICE: str = os.getenv("DEVICE", "auto")

# ---------------------------------------------------------------------------
# Logging / Experiment Tracking
# ---------------------------------------------------------------------------

ENABLE_LOGGING: bool = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "experiments/results_log.csv")

# ---------------------------------------------------------------------------
# Bloom Taxonomy Levels  (single source of truth)
# ---------------------------------------------------------------------------

BLOOM_LEVELS: list[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

BLOOM_DEFINITIONS: dict[str, str] = {
    "Remember": (
        "Recall facts, definitions, terms, formulas, or previously learned information."
    ),
    "Understand": (
        "Explain, interpret, summarize, describe, or demonstrate understanding of a concept."
    ),
    "Apply": (
        "Use learned knowledge, a method, rule, formula, or procedure to solve "
        "a problem or situation."
    ),
    "Analyze": (
        "Break information into parts, identify relationships, compare components, "
        "examine causes, or determine how parts relate to a whole."
    ),
    "Evaluate": (
        "Make a judgment, justify a decision, critique something, assess alternatives, "
        "or support a conclusion using criteria or evidence."
    ),
    "Create": (
        "Generate, design, formulate, construct, propose, or produce something new "
        "based on knowledge or reasoning."
    ),
}

# ---------------------------------------------------------------------------
# Research / UI Disclaimer
# ---------------------------------------------------------------------------

RESEARCH_DISCLAIMER: str = (
    "This system provides an AI-assisted Bloom's Taxonomy classification for research "
    "purposes. The result should be reviewed by an educator and should not be treated "
    "as a definitive assessment of a student's ability."
)
