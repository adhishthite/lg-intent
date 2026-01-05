"""
Base agent protocol and utilities.

Defines the interface that all retrieval agents must implement.
"""

from typing import Protocol

from enterprise_rag.state import RAGState, RetrievedChunk


class RetrievalAgent(Protocol):
    """
    Protocol for retrieval agents.

    All agents must implement a retrieve method that:
    - Takes the current state (primarily uses query)
    - Returns a list of RetrievedChunk objects

    This protocol is for documentation - actual agents are functions
    that return Command objects for LangGraph compatibility.
    """

    def retrieve(self, state: RAGState) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for the query."""
        ...
