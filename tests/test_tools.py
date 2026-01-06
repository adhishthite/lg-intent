"""Tests for search tools.

Tools accept focused queries and return retrieved chunks.
"""

import pytest

from enterprise_rag.search import close_clients
from enterprise_rag.tools import search_elastic_docs, search_internal_docs, search_jira


@pytest.fixture(autouse=True)
async def cleanup_es_clients():
    """Reset ES clients after each test."""
    yield
    await close_clients()


@pytest.mark.asyncio
async def test_search_internal_docs_returns_chunks():
    """Test that search_internal_docs returns chunks from wiki."""
    result = await search_internal_docs.ainvoke({"query": "PTO request"})

    assert len(result) > 0
    assert all("content" in chunk for chunk in result)
    assert all(chunk["source_type"] == "wiki" for chunk in result)


@pytest.mark.asyncio
async def test_search_elastic_docs_returns_chunks():
    """Test that search_elastic_docs returns chunks."""
    result = await search_elastic_docs.ainvoke({"query": "bool query"})

    assert len(result) > 0
    assert all(chunk["source_type"] == "elastic_docs" for chunk in result)


@pytest.mark.asyncio
async def test_search_jira_returns_chunks():
    """Test that search_jira returns chunks (mock data)."""
    result = await search_jira.ainvoke({"query": "search bug"})

    assert len(result) > 0
    assert all(chunk["source_type"] == "jira" for chunk in result)


@pytest.mark.asyncio
async def test_tools_accept_focused_queries():
    """Test that tools work with focused queries (not full multi-intent questions)."""
    # Each tool gets a focused query, not the full question
    internal_result = await search_internal_docs.ainvoke({"query": "PTO policy"})
    elastic_result = await search_elastic_docs.ainvoke({"query": "create index"})

    # Both should return relevant results for their focused queries
    assert len(internal_result) > 0
    assert len(elastic_result) > 0


@pytest.mark.asyncio
async def test_chunks_have_required_fields():
    """Test that all tool results have the required RetrievedChunk fields."""
    result = await search_internal_docs.ainvoke({"query": "test query"})

    for chunk in result:
        assert "content" in chunk
        assert "source_type" in chunk
        assert "source_url" in chunk
        assert "title" in chunk
        assert "relevance_score" in chunk
