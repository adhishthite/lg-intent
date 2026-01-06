"""
Main entry point for Enterprise RAG.

Run with a custom query:
    python src/main.py "What's the PTO policy?"

Or run demo queries:
    python src/main.py
"""

import asyncio
import sys

from enterprise_rag import create_rag_graph
from enterprise_rag.search import close_clients
from enterprise_rag.state import RAGState


async def run_query(graph, query: str) -> None:
    """
    Run a single query through the RAG system.

    Args:
        graph: Compiled RAG graph
        query: User's natural language query
    """
    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print(f"{'=' * 70}")

    # Initialize state with query (simplified for tool-based architecture)
    initial_state: RAGState = {
        "query": query,
        "retrieved_chunks": [],
        "final_chunks": [],
        "response": None,
        "sources": [],
    }

    # Stream execution to see each step (async)
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            print(f"\n[{node_name}]")

            # Show retrieved chunks count (from tool calls)
            if updates and updates.get("retrieved_chunks"):
                chunks = updates["retrieved_chunks"]
                print(f"  Retrieved {len(chunks)} chunks from tools")

                # Show source distribution
                source_counts: dict[str, int] = {}
                for chunk in chunks:
                    source = chunk.get("source_type", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                if source_counts:
                    dist = ", ".join(f"{k}: {v}" for k, v in source_counts.items())
                    print(f"  Source distribution: {dist}")

            # Show final chunks (post-rerank)
            if updates and updates.get("final_chunks"):
                chunks = updates["final_chunks"]
                print(f"  Reranked to {len(chunks)} final chunks:")
                for chunk in chunks[:5]:  # Show top 5
                    score = chunk.get("relevance_score", "N/A")
                    if isinstance(score, float):
                        score = f"{score:.3f}"
                    print(
                        f"    - [{chunk['source_type']}] {chunk.get('title', 'Untitled')[:40]} (score: {score})"
                    )
                if len(chunks) > 5:
                    print(f"    ... and {len(chunks) - 5} more")

            # Show response
            if updates and updates.get("response"):
                print(f"\n  Response:\n  {'-' * 60}")
                # Indent the response for readability
                response_lines = updates["response"].split("\n")
                for line in response_lines:
                    print(f"  {line}")

            # Show sources
            if updates and updates.get("sources"):
                print("\n  Sources:")
                for source in updates["sources"]:
                    url = source.get("url") or "No URL"
                    print(f"    - [{source['source_type']}] {source['title']}")
                    print(f"      {url}")


async def main():
    """
    Run the RAG system.

    If a query is provided as CLI argument, run that query.
    Otherwise, run example demo queries.
    """
    print("Creating Enterprise RAG graph (tool-based)...")
    graph = create_rag_graph()

    try:
        # Check for CLI argument
        if len(sys.argv) > 1:
            # Run user-provided query
            query = " ".join(sys.argv[1:])
            await run_query(graph, query)
        else:
            # Run demo queries showing different scenarios
            demo_queries = [
                # General query - no tools called
                "Hello!",
                # Single tool - internal docs
                "How do I submit a PTO request?",
                # Single tool - elastic docs
                "How do I create a bool query in Elasticsearch?",
                # Multi-tool - both internal and elastic docs
                "What's the PTO policy and how do I create an index?",
            ]
            for query in demo_queries:
                await run_query(graph, query)

    finally:
        # Clean up ES and embedding clients
        await close_clients()


if __name__ == "__main__":
    asyncio.run(main())
