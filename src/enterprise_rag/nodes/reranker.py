"""
Cross-source reranker node.

Reranks chunks from all tool calls using Jina cross-encoder.
"""

from enterprise_rag.search import rerank_chunks
from enterprise_rag.state import RAGState


async def reranker(state: RAGState) -> dict:
    """
    Rerank chunks from all sources using Jina cross-encoder.

    In the tool-based architecture, this node runs after the orchestrator
    has executed all tool calls. The state["retrieved_chunks"] field
    contains accumulated results from all tools.

    Pipeline:
    1. Collect all chunks from retrieved_chunks (from tool calls)
    2. Rerank using Jina cross-encoder for cross-source precision
    3. Store top results in final_chunks (used by draft_response)

    Args:
        state: Current RAG state with accumulated retrieved_chunks

    Returns:
        Dict with final_chunks after reranking
    """
    all_chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    if not all_chunks:
        return {"final_chunks": []}

    # Log source distribution (useful for debugging)
    source_counts: dict[str, int] = {}
    for chunk in all_chunks:
        source = chunk.get("source_type", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    # Jina rerank across all sources
    # This is the key step - cross-encoder evaluates query-document pairs
    # regardless of which agent retrieved them
    reranked = await rerank_chunks(query, all_chunks)

    return {"final_chunks": reranked}
