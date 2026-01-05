"""
Internal documentation agent.

Retrieves from internal wiki and ServiceNow knowledge base.
Currently returns mock data - swap with real retrieval later.
"""

from typing import Literal

from langgraph.types import Command

from enterprise_rag.state import RAGState, RetrievedChunk


def _mock_wiki_search(query: str) -> list[RetrievedChunk]:
    """Mock wiki search - replace with real implementation."""
    return [
        {
            "content": f"Wiki article about: {query}\n\n"
            "This is a mock wiki article. In production, this would be "
            "retrieved from your internal wiki (Confluence, Notion, etc.).\n\n"
            "Key points:\n"
            "- First important point\n"
            "- Second important point\n"
            "- Third important point",
            "source_type": "wiki",
            "source_url": "https://wiki.internal.example.com/article/123",
            "title": f"Internal Guide: {query[:30]}...",
            "relevance_score": 0.92,
        },
        {
            "content": "Related wiki content that provides additional context "
            "for the user's query. This demonstrates multiple chunks "
            "from the same source type.",
            "source_type": "wiki",
            "source_url": "https://wiki.internal.example.com/article/456",
            "title": "Related Policies and Procedures",
            "relevance_score": 0.78,
        },
    ]


def _mock_servicenow_search(query: str) -> list[RetrievedChunk]:
    """Mock ServiceNow search - replace with real implementation."""
    return [
        {
            "content": f"ServiceNow KB article for: {query}\n\n"
            "This knowledge base article explains the standard process.\n\n"
            "Steps:\n"
            "1. Navigate to the portal\n"
            "2. Fill out the required form\n"
            "3. Submit for approval\n"
            "4. Track status in your dashboard",
            "source_type": "servicenow",
            "source_url": "https://servicenow.example.com/kb/KB0001234",
            "title": "How To: Standard Process Guide",
            "relevance_score": 0.85,
        },
    ]


async def internal_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    """
    Retrieve from internal documentation sources.

    Searches both wiki and ServiceNow, combines results, and routes
    to draft_response for answer generation.

    Args:
        state: Current RAG state with query

    Returns:
        Command with retrieved_chunks update, routing to draft_response
    """
    query = state["query"]

    # Search both sources (mocks are sync, production will use async HTTP)
    wiki_chunks = _mock_wiki_search(query)
    servicenow_chunks = _mock_servicenow_search(query)

    # Combine and sort by relevance
    all_chunks = wiki_chunks + servicenow_chunks
    all_chunks.sort(key=lambda x: x.get("relevance_score") or 0, reverse=True)

    return Command(
        update={"retrieved_chunks": all_chunks},
        goto="draft_response",
    )
