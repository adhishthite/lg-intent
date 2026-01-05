"""Tests for retrieval agents."""

import pytest

from enterprise_rag.agents import elastic_docs_agent, internal_docs_agent, jira_agent
from enterprise_rag.state import RAGState


def _create_test_state(query: str) -> RAGState:
    """Create a test state with the given query."""
    return {
        "query": query,
        "intent": None,
        "classification_reasoning": None,
        "retrieved_chunks": [],
        "response": None,
        "sources": [],
    }


@pytest.mark.asyncio
async def test_internal_docs_agent_returns_chunks():
    """Test that internal_docs_agent returns chunks from wiki and servicenow."""
    state = _create_test_state("How do I submit a PTO request?")
    result = await internal_docs_agent(state)

    # Check that we got chunks
    chunks = result.update["retrieved_chunks"]
    assert len(chunks) > 0

    # Check source types
    source_types = {chunk["source_type"] for chunk in chunks}
    assert "wiki" in source_types
    assert "servicenow" in source_types

    # Check routing
    assert result.goto == "draft_response"


@pytest.mark.asyncio
async def test_elastic_docs_agent_returns_chunks():
    """Test that elastic_docs_agent returns chunks."""
    state = _create_test_state("How do I create an index?")
    result = await elastic_docs_agent(state)

    chunks = result.update["retrieved_chunks"]
    assert len(chunks) > 0
    assert all(chunk["source_type"] == "elastic_docs" for chunk in chunks)
    assert result.goto == "draft_response"


@pytest.mark.asyncio
async def test_jira_agent_returns_chunks():
    """Test that jira_agent returns chunks."""
    state = _create_test_state("What's the status of ISSUE-1234?")
    result = await jira_agent(state)

    chunks = result.update["retrieved_chunks"]
    assert len(chunks) > 0
    assert all(chunk["source_type"] == "jira" for chunk in chunks)
    assert result.goto == "draft_response"


@pytest.mark.asyncio
async def test_chunks_have_required_fields():
    """Test that all chunks have the required fields."""
    state = _create_test_state("test query")

    for agent in [internal_docs_agent, elastic_docs_agent, jira_agent]:
        result = await agent(state)
        for chunk in result.update["retrieved_chunks"]:
            assert "content" in chunk
            assert "source_type" in chunk
            assert "source_url" in chunk
            assert "title" in chunk
            assert "relevance_score" in chunk
