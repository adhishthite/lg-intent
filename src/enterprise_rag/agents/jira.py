"""
Jira agent.

Retrieves from Elastic Jira issue tracker.
Currently returns mock data - swap with real retrieval later.
"""

from typing import Literal

from langgraph.types import Command

from enterprise_rag.state import RAGState, RetrievedChunk


def _mock_jira_search(query: str) -> list[RetrievedChunk]:
    """Mock Jira search - replace with real implementation."""
    return [
        {
            "content": f"**ISSUE-1234**: Search feature bug\n\n"
            f"**Status**: In Progress\n"
            f"**Assignee**: Jane Developer\n"
            f"**Priority**: High\n\n"
            f"**Description**:\n"
            f"Related to query: {query}\n\n"
            f"The search feature is returning incorrect results when "
            f"special characters are used in the query string.\n\n"
            f"**Recent Comments**:\n"
            f"- @jane: Working on a fix, ETA end of week\n"
            f"- @bob: Confirmed reproduction steps",
            "source_type": "jira",
            "source_url": "https://jira.elastic.co/browse/ISSUE-1234",
            "title": "ISSUE-1234: Search feature bug",
            "relevance_score": 0.88,
        },
        {
            "content": "**ISSUE-5678**: Feature request - Advanced filters\n\n"
            "**Status**: Open\n"
            "**Assignee**: Unassigned\n"
            "**Priority**: Medium\n\n"
            "**Description**:\n"
            "Users have requested the ability to save and reuse filter configurations.\n\n"
            "**Acceptance Criteria**:\n"
            "- Users can save current filter state\n"
            "- Saved filters appear in a dropdown\n"
            "- Filters can be shared with team members",
            "source_type": "jira",
            "source_url": "https://jira.elastic.co/browse/ISSUE-5678",
            "title": "ISSUE-5678: Feature request - Advanced filters",
            "relevance_score": 0.72,
        },
    ]


def jira_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    """
    Retrieve from Jira issue tracker.

    Searches Jira for relevant issues and routes to draft_response
    for answer generation.

    Args:
        state: Current RAG state with query

    Returns:
        Command with retrieved_chunks update, routing to draft_response
    """
    query = state["query"]

    chunks = _mock_jira_search(query)

    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
