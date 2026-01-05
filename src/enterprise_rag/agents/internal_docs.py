"""
Internal documentation agent.

Retrieves from internal wiki using Elasticsearch hybrid search.
"""

from typing import Literal

from langgraph.types import Command

from enterprise_rag.config import settings
from enterprise_rag.search import hybrid_search
from enterprise_rag.state import RAGState


async def internal_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    """
    Retrieve from internal wiki documentation.

    Performs hybrid search (BM25 + vector with RRF) against the wiki index.

    Args:
        state: Current RAG state with query

    Returns:
        Command with retrieved_chunks update, routing to draft_response
    """
    query = state["query"]

    # Search wiki index
    chunks = await hybrid_search(
        index=settings.elasticsearch.WIKI_ES_VECTOR_INDEX,
        query=query,
        source_type="wiki",
    )

    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
