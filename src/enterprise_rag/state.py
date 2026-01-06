"""
State schemas for Enterprise RAG system.

This module defines the TypedDict schemas that flow through the LangGraph workflow.
Following the principle: store raw data, not formatted prompts.

Updated for tool-based architecture - intent classification is now implicit
in the orchestrator's tool selection.
"""

import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict

# =============================================================================
# Type Aliases
# =============================================================================

# Source types for retrieved chunks
SourceType = Literal["wiki", "servicenow", "elastic_docs", "jira"]


# =============================================================================
# Retrieved Data Schemas
# =============================================================================


class RetrievedChunk(TypedDict):
    """
    A single piece of retrieved context from any data source.

    This is the standard format that all tools return. Each tool
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
# Main State Schema
# =============================================================================


class RAGState(TypedDict):
    """
    The main state schema for the Enterprise RAG workflow (tool-based architecture).

    This TypedDict flows between all nodes in the graph. Each node
    reads what it needs and returns updates to specific fields.

    Design principles:
    - Store raw data, not formatted prompts
    - Each field has a clear owner (which node sets it)
    - Optional fields use | None pattern
    - Use Annotated with operator.add for accumulation

    Note: Intent classification fields (intents, classification_reasoning,
    query_keywords) have been removed. In the tool-based architecture,
    intent is implicit in the orchestrator's tool selection.
    """

    # -------------------------------------------------------------------------
    # INPUT (set at invocation)
    # -------------------------------------------------------------------------

    # The user's natural language query
    query: str

    # -------------------------------------------------------------------------
    # RETRIEVAL (set by orchestrator via tool calls)
    # -------------------------------------------------------------------------

    # Chunks retrieved by tools - accumulated from all tool calls
    # Uses operator.add reducer for accumulation
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add]

    # -------------------------------------------------------------------------
    # RERANKING (set by reranker node)
    # -------------------------------------------------------------------------

    # Final chunks after cross-source Jina reranking
    # Separate field because we can't "reset" a reducer field
    final_chunks: list[RetrievedChunk]

    # -------------------------------------------------------------------------
    # OUTPUT (set by draft_response or orchestrator for general queries)
    # -------------------------------------------------------------------------

    # The generated response to the user
    response: str | None

    # Sources cited in the response
    sources: list[Source]
