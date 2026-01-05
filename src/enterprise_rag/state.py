"""
State schemas for Enterprise RAG system.

This module defines the TypedDict schemas that flow through the LangGraph workflow.
Following the principle: store raw data, not formatted prompts.
"""

from typing import Literal

from typing_extensions import TypedDict

# =============================================================================
# Intent Types
# =============================================================================

IntentType = Literal["internal_docs", "elastic_docs", "jira"]

# Source types for retrieved chunks
SourceType = Literal["wiki", "servicenow", "elastic_docs", "jira"]


# =============================================================================
# Retrieved Data Schemas
# =============================================================================


class RetrievedChunk(TypedDict):
    """
    A single piece of retrieved context from any data source.

    This is the standard format that all agents return. Each agent
    populates these fields based on their source system.
    """

    # The actual text content retrieved
    content: str

    # Which source system this came from
    source_type: SourceType

    # URL to the original document (if available)
    source_url: str | None

    # Title of the document/article/issue
    title: str | None

    # Relevance score from the retrieval system (0.0 - 1.0)
    relevance_score: float | None


class Source(TypedDict):
    """
    Citation for the final response.

    These are extracted from RetrievedChunks and included in the
    response for transparency and verification.
    """

    title: str
    url: str | None
    source_type: SourceType


# =============================================================================
# Classification Schema
# =============================================================================


class IntentClassification(TypedDict):
    """
    Structured output from the intent classifier.

    Used with llm.with_structured_output() to get reliable classification.
    """

    # The classified intent - determines which agent handles the query
    intent: IntentType

    # Brief reasoning for the classification (useful for debugging)
    reasoning: str


# =============================================================================
# Main State Schema
# =============================================================================


class RAGState(TypedDict):
    """
    The main state schema for the Enterprise RAG workflow.

    This TypedDict flows between all nodes in the graph. Each node
    reads what it needs and returns updates to specific fields.

    Design principles:
    - Store raw data, not formatted prompts
    - Each field has a clear owner (which node sets it)
    - Optional fields use | None pattern
    """

    # ─────────────────────────────────────────────────────────────────────────
    # INPUT (set at invocation, read by multiple nodes)
    # ─────────────────────────────────────────────────────────────────────────

    # The user's natural language query
    query: str

    # ─────────────────────────────────────────────────────────────────────────
    # CLASSIFICATION (set by classify_intent node)
    # ─────────────────────────────────────────────────────────────────────────

    # The classified intent - determines routing
    intent: IntentType | None

    # Reasoning from the classifier (for debugging/logging)
    classification_reasoning: str | None

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVAL (set by agent nodes)
    # ─────────────────────────────────────────────────────────────────────────

    # Chunks retrieved by the selected agent
    # This is a list that the agent populates
    retrieved_chunks: list[RetrievedChunk]

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT (set by draft_response node)
    # ─────────────────────────────────────────────────────────────────────────

    # The generated response to the user
    response: str | None

    # Sources cited in the response
    sources: list[Source]
