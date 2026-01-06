"""Tests for state schemas (tool-based architecture)."""

from enterprise_rag.state import (
    RAGState,
    RetrievedChunk,
    Source,
)


def test_retrieved_chunk_structure():
    """Test that RetrievedChunk has expected fields."""
    chunk: RetrievedChunk = {
        "content": "Test content",
        "source_type": "wiki",
        "source_url": "https://example.com",
        "title": "Test Title",
        "relevance_score": 0.95,
    }
    assert chunk["content"] == "Test content"
    assert chunk["source_type"] == "wiki"
    assert chunk["relevance_score"] == 0.95


def test_source_structure():
    """Test that Source has expected fields."""
    source: Source = {
        "title": "Test Source",
        "url": "https://example.com",
        "source_type": "elastic_docs",
    }
    assert source["title"] == "Test Source"
    assert source["source_type"] == "elastic_docs"


def test_rag_state_structure():
    """Test that RAGState has expected fields (tool-based architecture)."""
    # Simplified state - no longer has intents, classification_reasoning, query_keywords
    state: RAGState = {
        "query": "How do I reset my password?",
        "retrieved_chunks": [],
        "final_chunks": [],
        "response": None,
        "sources": [],
    }
    assert state["query"] == "How do I reset my password?"
    assert state["retrieved_chunks"] == []
    assert state["final_chunks"] == []
    assert state["response"] is None


def test_rag_state_with_chunks():
    """Test RAGState with retrieved chunks."""
    chunk: RetrievedChunk = {
        "content": "Password reset instructions...",
        "source_type": "wiki",
        "source_url": "https://wiki.example.com/password",
        "title": "Password Reset Guide",
        "relevance_score": 0.92,
    }
    state: RAGState = {
        "query": "How do I reset my password?",
        "retrieved_chunks": [chunk],
        "final_chunks": [chunk],
        "response": "Here's how to reset your password...",
        "sources": [
            {
                "title": "Password Reset Guide",
                "url": "https://wiki.example.com/password",
                "source_type": "wiki",
            }
        ],
    }
    assert len(state["retrieved_chunks"]) == 1
    assert len(state["final_chunks"]) == 1
    assert state["response"] is not None
    assert len(state["sources"]) == 1
