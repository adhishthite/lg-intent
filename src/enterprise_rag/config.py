"""
Configuration for Enterprise RAG system.

Uses environment variables for configuration, with sensible defaults.
"""

import os

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


# =============================================================================
# LLM Configuration
# =============================================================================

# Model for intent classification (fast, cheap)
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-5-nano")

# Model for response generation (more capable)
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "gpt-5-nano")

# Temperature for classification (deterministic)
CLASSIFIER_TEMPERATURE = 0.0

# Temperature for response generation (slight creativity)
RESPONSE_TEMPERATURE = 0.3


# =============================================================================
# Retrieval Configuration
# =============================================================================

# Maximum chunks to retrieve per agent
MAX_CHUNKS_PER_AGENT = int(os.getenv("MAX_CHUNKS_PER_AGENT", "5"))

# Minimum relevance score to include a chunk
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))


# =============================================================================
# Feature Flags
# =============================================================================

# Enable debug logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
