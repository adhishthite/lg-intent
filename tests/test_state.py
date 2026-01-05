"""Tests for state schemas."""

from enterprise_rag.state import (
    IntentClassification,
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


def test_intent_classification_structure():
    """Test that IntentClassification has expected fields."""
    classification: IntentClassification = {
        "intent": "internal_docs",
        "reasoning": "User is asking about internal process",
    }
    assert classification["intent"] == "internal_docs"
    assert "internal" in classification["reasoning"]


def test_rag_state_structure():
    """Test that RAGState has expected fields."""
    state: RAGState = {
        "query": "How do I reset my password?",
        "intent": None,
        "classification_reasoning": None,
        "retrieved_chunks": [],
        "response": None,
        "sources": [],
    }
    assert state["query"] == "How do I reset my password?"
    assert state["intent"] is None
    assert state["retrieved_chunks"] == []
