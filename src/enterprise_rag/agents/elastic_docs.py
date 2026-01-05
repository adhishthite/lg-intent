"""
Elasticsearch documentation agent.

Retrieves from Elasticsearch official documentation using hybrid search.
"""

from typing import Literal

from langgraph.types import Command

from enterprise_rag.config import settings
from enterprise_rag.search import hybrid_search
from enterprise_rag.state import RAGState


async def elastic_docs_agent(state: RAGState) -> Command[Literal["draft_response"]]:
    """
    Retrieve from Elasticsearch official documentation.

    Performs hybrid search (BM25 + vector with RRF) against the docs index.

    Args:
        state: Current RAG state with query

    Returns:
        Command with retrieved_chunks update, routing to draft_response
    """
    query = state["query"]

    # Search docs index
    chunks = await hybrid_search(
        index=settings.elasticsearch.DOCS_ES_VECTOR_INDEX,
        query=query,
        source_type="elastic_docs",
    )

    return Command(
        update={"retrieved_chunks": chunks},
        goto="draft_response",
    )
