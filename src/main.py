"""
Main entry point for Enterprise RAG.

Demonstrates how to invoke the RAG graph with sample queries.
"""

import asyncio

from enterprise_rag import create_rag_graph
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

    # Initialize state with query and empty collections
    initial_state: RAGState = {
        "query": query,
        "intent": None,
        "classification_reasoning": None,
        "retrieved_chunks": [],
        "response": None,
        "sources": [],
    }

    # Stream execution to see each step (async)
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            print(f"\n[{node_name}]")

            # Show classification info
            if updates and updates.get("intent"):
                print(f"  Intent: {updates['intent']}")
                if updates.get("classification_reasoning"):
                    print(f"  Reasoning: {updates['classification_reasoning']}")

            # Show retrieved chunks count
            if updates and updates.get("retrieved_chunks"):
                chunks = updates["retrieved_chunks"]
                print(f"  Retrieved {len(chunks)} chunks:")
                for chunk in chunks:
                    score = chunk.get("relevance_score", "N/A")
                    print(
                        f"    - [{chunk['source_type']}] {chunk.get('title', 'Untitled')} (score: {score})"
                    )

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
    """Run example queries to demonstrate the RAG system."""
    print("Creating Enterprise RAG graph...")
    graph = create_rag_graph()

    # ─────────────────────────────────────────────────────────────────────────
    # Example 1: Internal documentation query
    # ─────────────────────────────────────────────────────────────────────────
    await run_query(
        graph,
        "How do I submit a PTO request?",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Example 2: Elasticsearch documentation query
    # ─────────────────────────────────────────────────────────────────────────
    await run_query(
        graph,
        "How do I create a bool query in Elasticsearch?",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Example 3: Jira query
    # ─────────────────────────────────────────────────────────────────────────
    await run_query(
        graph,
        "What's the status of the search feature bug?",
    )


if __name__ == "__main__":
    asyncio.run(main())
